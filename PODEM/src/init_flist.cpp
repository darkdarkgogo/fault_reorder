/**********************************************************************/
/*           Build the fault list;                                    */
/*           Calculate the total fault coverage.                      */
/*           Building the dummy input and output gate list;           */
/*                                                                    */
/*           Author: Bing-Chen (Benson) Wu                            */
/*           update: 01/21/2018                                       */
/*           Revised: Cheng-Yun Hsieh                                 */
/*           modified:02/05/2020   revise fault collapsing            */
/**********************************************************************/

#include "atpg.h"
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

void require_fault_map(bool condition, const string &message) {
  if (!condition) throw runtime_error(message);
}

string fnv1a_file_hash(const string &path) {
  ifstream input(path.c_str(), ios::binary);
  require_fault_map(input.good(), "Cannot hash BENCH file: " + path);
  uint64_t value = UINT64_C(14695981039346656037);
  char byte;
  while (input.get(byte)) {
    value ^= static_cast<unsigned char>(byte);
    value *= UINT64_C(1099511628211);
  }
  char result[17];
  snprintf(result, sizeof(result), "%016llx",
           static_cast<unsigned long long>(value));
  return result;
}

} // namespace

void ATPG::generate_fault_list() {
  if (!fault_map_path.empty()) {
    load_mapped_fault_list();
    if (fault_order_by_scoap) {
      calculate_scoap();
      fault_reorder();
    }
    return;
  }
  int fault_num;
  wptr w;
  nptr n;
  fptr_s f;
  unordered_map<wptr, int> num_of_eqv_sa0, num_of_eqv_sa1;
  /* walk through every wire in the circuit*/
  for (auto pos : sort_wlist) {
    w = pos;
    n = w->inode.front();

    /* for each gate, create a gate output stuck-at zero (SA0) fault */
    f = move(fptr_s(new(nothrow) FAULT));
    if (f == nullptr) error("No more room!");
    f->node = n;

    f->io = GO;     // gate output SA0 fault
    f->fault_type = STUCK0;
    f->to_swlist = w->wlist_index;
    // for AND NOR NOT BUF, their GI fault is equivalent to GO SA0 fault
    // For each fault f, we count how many faults are in the same equivalent class as f.
    //   So that, we can calculate uncollapsed fault coverage.

    f->eqv_fault_num = 1;    // GO SA0 fault itself
    switch (n->type) {
      case AND:
      case BUF:
        for (wptr wptr_ele: w->inode.front()->iwire) {
          f->eqv_fault_num += num_of_eqv_sa0[wptr_ele];   // count how many equivalent faults in gate input
        }
        break;
      case NOR:
      case NOT:
        for (wptr wptr_ele: w->inode.front()->iwire) {
          f->eqv_fault_num += num_of_eqv_sa1[wptr_ele];
        }
        break;
      case INPUT:
      case OR:
      case NAND:
      case EQV:
      case XOR:
        break;
    }
    //  these are representative fault in the equivalent fault class
    //  For a wire w,  w.onode is a vector of w's fanout nodes.
    //   w->onode.size is the number of fanout nodes.  if w->onode.size >1, w is a fanout stem.
    //   w->onode.front()->type is the gate type at the output of w (assume w has only 1 fanout)
    //   w->onode.front()->type = OR means w is an input of OR gate.
    if (w->onode.size() > 1 || w->onode.front()->type == EQV || w->onode.front()->type == OR
        || w->onode.front()->type == NOR || w->onode.front()->type == XOR || w->onode.front()->type == OUTPUT) {
      num_of_gate_fault += f->eqv_fault_num; // accumulate total uncollapsed faults
      flist_undetect.push_front(f.get()); // initial undetected fault list contains all faults
      flist.push_front(move(f));  // push into the fault list
    } else {
      num_of_eqv_sa0[w] = f->eqv_fault_num;
    }

    /* for each gate, create a gate output stuck-at one (SA1) fault */
    f = move(fptr_s(new(nothrow) FAULT));

    if (f == nullptr) error("No more room!");
    f->eqv_fault_num = 1;
    f->node = n;
    f->io = GO;
    f->fault_type = STUCK1;
    f->to_swlist = w->wlist_index;
    /* for OR NAND NOT BUF, their GI fault is equivalent to GO SA1 fault */
    switch (n->type) {
      case OR:
      case BUF:
        for (wptr wptr_ele: w->inode.front()->iwire) {
          if (num_of_eqv_sa1.find(wptr_ele) == num_of_eqv_sa1.end())
            cerr << wptr_ele << " is not in hashmap." << endl;
          f->eqv_fault_num += num_of_eqv_sa1[wptr_ele];
        }
        break;
      case NAND:
      case NOT:
        for (wptr wptr_ele: w->inode.front()->iwire) {
          if (num_of_eqv_sa0.find(wptr_ele) == num_of_eqv_sa0.end())
            cerr << wptr_ele << " is not in hashmap." << endl;
          f->eqv_fault_num += num_of_eqv_sa0[wptr_ele];
        }
        break;
      case INPUT:
      case AND:
      case NOR:
      case EQV:
      case XOR:
        break;
    }
    if (w->onode.size() > 1 || w->onode.front()->type == EQV || w->onode.front()->type == AND
        || w->onode.front()->type == NAND || w->onode.front()->type == XOR || w->onode.front()->type == OUTPUT) {
      num_of_gate_fault += f->eqv_fault_num; // accumulate total fault count
      flist_undetect.push_front(f.get()); // initial undetected fault list contains all faults
      flist.push_front(move(f));  // push into the fault list
    } else {
      num_of_eqv_sa1[w] = f->eqv_fault_num;
    }

    /*if w has multiple fanout branches */
    if (w->onode.size() > 1) {
      num_of_eqv_sa0[w] = num_of_eqv_sa1[w] = 1;
      for (nptr nptr_ele: w->onode) {
        /* create SA0 for OR NOR EQV XOR gate inputs  */
        switch (nptr_ele->type) {
          case OUTPUT:
          case OR:
          case NOR:
          case EQV:
          case XOR:
            f = move(fptr_s(new(nothrow) FAULT));
            if (f == nullptr) error("No more room!");
            f->node = nptr_ele;
            f->io = GI;
            f->fault_type = STUCK0;
            f->to_swlist = w->wlist_index;
            f->eqv_fault_num = 1;
            /* f->index is the index number of gate input,
               which GI fault is associated with*/
            for (int k = 0; k < nptr_ele->iwire.size(); k++) {
              if (nptr_ele->iwire[k] == w) f->index = k;
            }
            num_of_gate_fault++;
            flist_undetect.push_front(f.get());
            flist.push_front(move(f));
            break;
        }
        /* create SA1 for AND NAND EQV XOR gate inputs  */
        switch (nptr_ele->type) {
          case OUTPUT:
          case AND:
          case NAND:
          case EQV:
          case XOR:
            f = move(fptr_s(new(nothrow) FAULT));
            if (f == nullptr) error("No more room!");
            f->node = nptr_ele;
            f->io = GI;
            f->fault_type = STUCK1;
            f->to_swlist = w->wlist_index;
            f->eqv_fault_num = 1;
            for (int k = 0; k < nptr_ele->iwire.size(); k++) {
              if (nptr_ele->iwire[k] == w) f->index = k;
            }
            num_of_gate_fault++;
            flist_undetect.push_front(f.get());
            flist.push_front(move(f));
            break;
        }
      }
    }
  }
  flist.reverse();
  flist_undetect.reverse();
  /*walk through all faults, assign fault_no one by one  */
  fault_num = 0;
  for (fptr f: flist_undetect) {
    f->fault_no = fault_num;
    fault_num++;
    //cout << f->fault_no << f->node->name << ":" << (f->io?"O":"I") << (f->io?9:(f->index)) << "SA" << f->fault_type << endl;
  }

  if (fault_order_by_scoap) {
    calculate_scoap();
    fault_reorder();
  }

  unordered_map<string, int> identifier_counts;
  for (fptr fault : flist_undetect) {
    ++identifier_counts[fault_identifier(fault)];
  }
  unordered_map<string, int> duplicate_indices;
  for (fptr fault : flist_undetect) {
    const string identifier = fault_identifier(fault);
    if (identifier_counts[identifier] <= 1) continue;
    const int duplicate_index = duplicate_indices[identifier]++;
    if (duplicate_index > 0) {
      fault->external_id = identifier + "@dup" + to_string(duplicate_index);
    }
  }

  fprintf(stdout, "#number of equivalent faults = %d\n", fault_num);
}/* end of generate_fault_list */

string ATPG::fault_identifier(const fptr fault) const {
  if (!fault->external_id.empty()) return fault->external_id;
  string identifier = fault->node->name;
  if (fault->io == GO) {
    identifier += ":GO:";
  } else {
    identifier += ":GI" + to_string(fault->index) + ":";
  }
  identifier += fault->fault_type == STUCK1 ? "sa1" : "sa0";
  return identifier;
}

vector<ATPG::FaultCatalogEntry> ATPG::get_fault_catalog() const {
  vector<FaultCatalogEntry> result;
  for (const fptr fault : flist_undetect) {
    FaultCatalogEntry entry;
    entry.fault_id = fault_identifier(fault);
    entry.node_name = fault->node->name;
    entry.input_wire_name = fault->logical_xor_input
        ? fault->logical_input_wire->name
        : (fault->io == GI ? fault->node->iwire[fault->index]->name : "-");
    entry.io = fault->io;
    entry.input_index = fault->logical_xor_input
        ? fault->logical_input_index
        : (fault->io == GI ? fault->index : -1);
    entry.input_occurrence = -1;
    if (fault->logical_xor_input) {
      entry.input_occurrence = fault->logical_input_occurrence;
    } else if (fault->io == GI) {
      entry.input_occurrence = 0;
      for (int index = 0; index < fault->index; ++index) {
        if (fault->node->iwire[index]->name == entry.input_wire_name) {
          ++entry.input_occurrence;
        }
      }
    }
    entry.fault_type = fault->fault_type;
    entry.eqv_fault_num = fault->eqv_fault_num;
    result.push_back(entry);
  }
  return result;
}

void ATPG::load_mapped_fault_list() {
  ifstream input(fault_map_path.c_str());
  require_fault_map(input.good(), "Cannot open fault map: " + fault_map_path);

  string token;
  require_fault_map(static_cast<bool>(input >> token) &&
                        (token == "SMARTATPG_FAULT_MAP_V2" ||
                         token == "SMARTATPG_FAULT_MAP_V3"),
                    "Unsupported fault map format: " + fault_map_path);
  const bool has_logical_xor_inputs = token == "SMARTATPG_FAULT_MAP_V3";
  string source_hash;
  string circuit_hash;
  size_t count = 0;
  int expected_uncollapsed = 0;
  require_fault_map(static_cast<bool>(input >> token >> source_hash) &&
                        token == "source_hash" && source_hash.size() == 16 &&
                        all_of(source_hash.begin(), source_hash.end(),
                               [](unsigned char value) { return isxdigit(value) != 0; }),
                    "Invalid source hash in fault map: " + fault_map_path);
  require_fault_map(static_cast<bool>(input >> token >> circuit_hash) &&
                        token == "circuit_hash",
                    "Invalid circuit hash in fault map: " + fault_map_path);
  require_fault_map(circuit_hash == fnv1a_file_hash(filename),
                    "Fault map circuit hash does not match: " + filename);
  require_fault_map(static_cast<bool>(input >> token >> count) &&
                        token == "count" && count > 0,
                    "Invalid fault count in fault map: " + fault_map_path);
  require_fault_map(static_cast<bool>(input >> token >> expected_uncollapsed) &&
                        token == "uncollapsed_total" && expected_uncollapsed > 0,
                    "Invalid uncollapsed count in fault map: " + fault_map_path);

  unordered_set<string> fault_ids;
  int uncollapsed_total = 0;
  for (size_t fault_no = 0; fault_no < count; ++fault_no) {
    string external_id;
    string node_name;
    string input_wire_name;
    int io = -1;
    int mapped_input_occurrence = -1;
    int fault_type = -1;
    int eqv_fault_num = 0;
    int logical_xor_input = 0;
    int logical_input_index = -1;
    require_fault_map(
        static_cast<bool>(input >> token >> external_id >> node_name >>
                          input_wire_name >> io >> mapped_input_occurrence >>
                          fault_type >> eqv_fault_num) && token == "fault",
        "Truncated fault map record in: " + fault_map_path);
    if (has_logical_xor_inputs) {
      require_fault_map(static_cast<bool>(input >> logical_xor_input >> logical_input_index),
                        "Truncated V3 fault map record in: " + fault_map_path);
      require_fault_map(
          (logical_xor_input == 0 && logical_input_index == -1) ||
              (logical_xor_input == 1 && logical_input_index >= 0 &&
               logical_input_index < 2),
          "Invalid V3 logical XOR marker for: " + external_id);
    }
    require_fault_map(fault_ids.insert(external_id).second,
                      "Duplicate fault ID in fault map: " + external_id);
    require_fault_map((io == GI || io == GO) &&
                          (fault_type == STUCK0 || fault_type == STUCK1) &&
                          eqv_fault_num > 0,
                      "Invalid fault attributes for: " + external_id);

    nptr node = nfind(node_name);
    require_fault_map(node != nullptr, "Unknown mapped fault node: " + node_name);
    int input_index = 0;
    wptr target_wire = nullptr;
    wptr logical_input_wire = nullptr;
    if (logical_xor_input != 0) {
      require_fault_map(logical_xor_input == 1 && io == GI &&
                            logical_input_index >= 0 && mapped_input_occurrence >= 0 &&
                            node->type == NAND && !node->owire.empty(),
                        "Invalid logical XOR input fault: " + external_id);
      logical_input_wire = wfind(input_wire_name);
      require_fault_map(logical_input_wire != nullptr,
                        "Unknown logical XOR input wire: " + input_wire_name);
      target_wire = logical_input_wire;
    } else if (io == GI) {
      require_fault_map(mapped_input_occurrence >= 0,
                        "Mapped input occurrence is invalid on node " + node_name);
      int occurrence = 0;
      bool found_input = false;
      for (size_t index = 0; index < node->iwire.size(); ++index) {
        if (node->iwire[index]->name != input_wire_name) continue;
        if (occurrence++ == mapped_input_occurrence) {
          input_index = static_cast<int>(index);
          target_wire = node->iwire[index];
          found_input = true;
          break;
        }
      }
      require_fault_map(found_input,
                        "Unknown mapped input wire '" + input_wire_name +
                            "' on node " + node_name);
    } else {
      require_fault_map(input_wire_name == "-" && mapped_input_occurrence == -1,
                        "GO fault must use '-' and index -1");
      require_fault_map(!node->owire.empty(),
                        "Mapped GO node has no output wire: " + node_name);
      target_wire = node->owire.front();
    }

    fptr_s fault(new(nothrow) FAULT);
    if (fault == nullptr) error("No more room!");
    fault->node = node;
    fault->io = static_cast<short>(io);
    fault->index = static_cast<short>(input_index);
    fault->fault_type = static_cast<short>(fault_type);
    fault->eqv_fault_num = eqv_fault_num;
    fault->to_swlist = target_wire->wlist_index;
    fault->fault_no = static_cast<int>(fault_no);
    fault->external_id = external_id;
    fault->logical_xor_input = logical_xor_input != 0;
    fault->logical_input_wire = logical_input_wire;
    fault->logical_input_index = static_cast<short>(logical_input_index);
    fault->logical_input_occurrence = static_cast<short>(mapped_input_occurrence);
    uncollapsed_total += eqv_fault_num;
    flist_undetect.push_front(fault.get());
    flist.push_front(move(fault));
  }
  require_fault_map(static_cast<bool>(input >> token) && token == "end",
                    "Fault map is missing its end marker: " + fault_map_path);
  require_fault_map(!(input >> token),
                    "Unexpected trailing data in fault map: " + fault_map_path);
  require_fault_map(uncollapsed_total == expected_uncollapsed,
                    "Fault map equivalent-fault total does not match header");
  flist.reverse();
  flist_undetect.reverse();
  num_of_gate_fault = expected_uncollapsed;
  fprintf(stdout, "#number of equivalent faults = %zu\n", count);
}

/* computing the actual fault coverage */
void ATPG::compute_fault_coverage() {
  double gate_fault_coverage, eqv_gate_fault_coverage;
  int no_of_detect, eqv_no_of_detect, eqv_num_of_gate_fault;
  fptr f;

  debug = 0;
  no_of_detect = 0;
  gate_fault_coverage = 0.0;
  eqv_no_of_detect = 0;
  eqv_gate_fault_coverage = 0.0;
  eqv_num_of_gate_fault = 0;

  for (const auto &pos : flist) {
    f = pos.get();
    if (debug) {
      if (f->detect != 0) {
        switch (f->node->type) {
          case INPUT:
            fprintf(stdout, "primary input: %s\n", f->node->owire.front()->name.c_str());
            break;
          case OUTPUT:
            fprintf(stdout, "primary output: %s\n", f->node->iwire.front()->name.c_str());
            break;
          default:
            fprintf(stdout, "gate: %s ;", f->node->name.c_str());
            if (f->io == GI) {
              fprintf(stdout, "input wire name: %s\n",
                      (f->logical_xor_input ? f->logical_input_wire->name
                                            : f->node->iwire[f->index]->name).c_str());
            } else {
              fprintf(stdout, "output wire name: %s\n", f->node->owire.front()->name.c_str());
            }
            break;
        }
        fprintf(stdout, "fault_type = ");
        switch (f->fault_type) {
          case STUCK0:
            fprintf(stdout, "s-a-0\n");
            break;
          case STUCK1:
            fprintf(stdout, "s-a-1\n");
            break;
        }
        fprintf(stdout, "no of equivalent fault = %d\n", f->eqv_fault_num);
        fprintf(stdout, "detection flag = %d\n", f->detect);
        fprintf(stdout, "\n");
      }
    }
    if (f->detect == TRUE) {
      no_of_detect += f->eqv_fault_num;
      eqv_no_of_detect++;
    }
    eqv_num_of_gate_fault++;
  }
  if (num_of_gate_fault != 0)
    gate_fault_coverage = (((double) no_of_detect) / num_of_gate_fault) * 100;
  if (eqv_num_of_gate_fault != 0)
    eqv_gate_fault_coverage = (((double) eqv_no_of_detect) / eqv_num_of_gate_fault) * 100;

  /* print out fault coverage results */
  fprintf(stdout, "\n");
  fprintf(stdout, "#FAULT COVERAGE RESULTS :\n");
  fprintf(stdout, "#number of test vectors = %d\n", in_vector_no);
  fprintf(stdout,
          "#total number of gate faults (uncollapsed) = %d\n",
          num_of_gate_fault);  // uncollapsed gate-level fault
  fprintf(stdout, "#total number of detected faults = %d\n", no_of_detect);
  fprintf(stdout, "#total gate fault coverage = %5.2f%%\n", gate_fault_coverage);  // uncollapsed fault coverage
  fprintf(stdout, "#number of equivalent gate faults (collapsed) = %d\n", eqv_num_of_gate_fault);
  fprintf(stdout, "#number of equivalent detected faults = %d\n", eqv_no_of_detect);
  fprintf(stdout, "#equivalent gate fault coverage = %5.2f%%\n", eqv_gate_fault_coverage);
  fprintf(stdout, "\n");
}/* end of compute_fault_coverage */

/* for each PI and PO in the whole circuit,
   create a dummy PI gate to feed each PI wire 
   create a dummy PO gate to feed each PO wire. 
   The reason we need dummy gate is to help to deal with boundary conditions easily.
   */

//
// for a primary input, we have this schematic
//   node1 (dummy PI gate) - wireA - node2 (gate)
// we have two fault on wireA, associated with "node1 output faults" 
//
// for a primary output, we have this schematic
//   node1 (gate) - wireA - node2 (dummy PO gate) 
// we have two fault on wireA, associated with "node1 output faults" 
//

void ATPG::create_dummy_gate() {
  int i;
  int num_of_dummy;
  nptr n;
  char sgate[25];
  char intstring[25];

  num_of_dummy = 0;

  /* create a dummy PI gate for each PI wire */
  for (i = 0; i < cktin.size(); i++) {
    num_of_dummy++;

    /* the dummy gate name is  "dummay_gate#"  */
    sprintf(intstring, "%d", num_of_dummy);
    sprintf(sgate, "dummy_gate%s", intstring);

    /* n is the dummy PI gate, cktin[i]is the original PI wire.  feed n to cktin[i] */
    n = getnode(string(sgate));
    n->type = INPUT;
    n->owire.push_back(cktin[i]);
    cktin[i]->inode.push_back(n);
  } // for i

  /* create a dummy PO gate for each PO wire */
  for (i = 0; i < cktout.size(); i++) {
    num_of_dummy++;

    /* the dummy gate name is  "dummay_gate#"  */
    sprintf(intstring, "%d", num_of_dummy);
    sprintf(sgate, "dummy_gate%s", intstring);

    /* n is the dummy PO gate, cktout[i] is the original PO wire.  feed cktout[i] to n */
    n = getnode(sgate);
    n->type = OUTPUT;
    n->iwire.push_back(cktout[i]);
    cktout[i]->onode.push_back(n);
  } // for i
}/* end of create_dummy_gate */

char ATPG::itoc(const int &i) {
  switch (i) {
    case 1:
      return '1';
    case 2:
      return 'U';
    case 0:
      return '0';
  }
}
