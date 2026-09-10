#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "atpg.h"

namespace py = pybind11;

namespace {

py::dict catalog_stuck_at(const std::string &circuit_path,
                          const std::string &fault_map_path) {
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_fault_map_path(fault_map_path);

  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
  }

  py::list faults;
  for (const ATPG::FaultCatalogEntry &entry : atpg.get_fault_catalog()) {
    py::dict item;
    item["fault_id"] = entry.fault_id;
    item["node_name"] = entry.node_name;
    item["input_wire_name"] = entry.input_wire_name;
    item["io"] = entry.io;
    item["input_index"] = entry.input_index;
    item["input_occurrence"] = entry.input_occurrence;
    item["fault_type"] = entry.fault_type;
    item["eqv_fault_num"] = entry.eqv_fault_num;
    faults.append(item);
  }

  py::dict result;
  result["faults"] = faults;
  result["uncollapsed_total"] = atpg.get_uncollapsed_fault_count();
  return result;
}

py::dict run_stuck_at_ordered(const std::string &circuit_path,
                              const std::string &fault_map_path,
                              const std::vector<std::string> &ordered_fault_ids,
                              int backtrack_limit, int seed) {
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_fault_map_path(fault_map_path);
  atpg.configure_ordered_stuck_at(backtrack_limit, seed);

  ATPG::AtpgRunResult summary;
  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    atpg.reorder_stuck_at_faults(ordered_fault_ids);
    summary = atpg.run_stuck_at(false);
  }

  py::dict result;
  result["pattern_count"] = summary.pattern_count;
  result["detected_collapsed_faults"] = summary.detected_collapsed_faults;
  result["detected_equivalent_faults"] = summary.detected_equivalent_faults;
  result["uncollapsed_faults"] = summary.uncollapsed_faults;
  result["aborted_faults"] = summary.aborted_faults;
  result["redundant_faults"] = summary.redundant_faults;
  result["podem_calls"] = summary.podem_calls;
  result["total_backtracks"] = summary.total_backtracks;
  return result;
}

} // namespace

PYBIND11_MODULE(cpp_podem, module) {
  module.doc() = "Minimal Python bridge for PODEM stuck-at fault catalogs";
  module.def("catalog_stuck_at", &catalog_stuck_at,
             py::arg("circuit_path"), py::arg("fault_map_path") = "");
  module.def("run_stuck_at_ordered", &run_stuck_at_ordered,
             py::arg("circuit_path"), py::arg("fault_map_path"),
             py::arg("ordered_fault_ids"), py::arg("backtrack_limit") = 3000,
             py::arg("seed") = 14);
}
