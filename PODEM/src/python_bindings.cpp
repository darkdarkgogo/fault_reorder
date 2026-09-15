#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "atpg.h"

namespace py = pybind11;

namespace {

py::dict catalog_to_dict(
    const std::vector<ATPG::FaultCatalogEntry> &catalog,
    int uncollapsed_total) {
  py::list faults;
  for (const ATPG::FaultCatalogEntry &entry : catalog) {
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
  result["uncollapsed_total"] = uncollapsed_total;
  return result;
}

py::dict summary_to_dict(const ATPG::AtpgRunResult &summary) {
  py::dict result;
  result["pattern_count"] = summary.pattern_count;
  result["detected_collapsed_faults"] = summary.detected_collapsed_faults;
  result["detected_equivalent_faults"] = summary.detected_equivalent_faults;
  result["uncollapsed_faults"] = summary.uncollapsed_faults;
  result["aborted_faults"] = summary.aborted_faults;
  result["redundant_faults"] = summary.redundant_faults;
  result["redundant_equivalent_faults"] = summary.redundant_equivalent_faults;
  result["podem_calls"] = summary.podem_calls;
  result["total_backtracks"] = summary.total_backtracks;
  return result;
}

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

  return catalog_to_dict(
      atpg.get_fault_catalog(), atpg.get_uncollapsed_fault_count());
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

  return summary_to_dict(summary);
}

class StuckAtSession {
public:
  StuckAtSession(const std::string &circuit_path,
                 const std::string &fault_map_path,
                 int backtrack_limit, int seed) {
    atpg_.detected_num = 1;
    atpg_.set_fault_map_path(fault_map_path);
    atpg_.configure_ordered_stuck_at(backtrack_limit, seed);
    py::gil_scoped_release release;
    atpg_.input(circuit_path);
    atpg_.level_circuit();
    atpg_.rearrange_gate_inputs();
    atpg_.create_dummy_gate();
    atpg_.generate_fault_list();
    catalog_ = atpg_.get_fault_catalog();
    uncollapsed_total_ = atpg_.get_uncollapsed_fault_count();
  }

  py::dict catalog() const {
    return catalog_to_dict(catalog_, uncollapsed_total_);
  }

  std::vector<std::string> remaining_fault_ids() const {
    return atpg_.get_selectable_fault_ids();
  }

  py::dict step(const std::string &fault_id) {
    ATPG::AtpgStepResult step_result;
    {
      py::gil_scoped_release release;
      step_result = atpg_.step_stuck_at(fault_id);
    }
    py::dict result = summary_to_dict(step_result.cumulative_result);
    result["selected_fault_id"] = step_result.selected_fault_id;
    result["target_status"] = step_result.target_status;
    result["generated_pattern"] = step_result.generated_pattern;
    result["newly_detected_fault_ids"] = step_result.newly_detected_fault_ids;
    result["remaining_fault_ids"] = step_result.remaining_fault_ids;
    result["current_pattern_count"] =
        step_result.cumulative_result.pattern_count;
    result["current_podem_calls"] =
        step_result.cumulative_result.podem_calls;
    result["current_total_backtracks"] =
        step_result.cumulative_result.total_backtracks;
    return result;
  }

  py::dict result() const {
    return summary_to_dict(atpg_.get_stuck_at_result());
  }

private:
  ATPG atpg_;
  std::vector<ATPG::FaultCatalogEntry> catalog_;
  int uncollapsed_total_{};
};

} // namespace

PYBIND11_MODULE(cpp_podem, module) {
  module.doc() = "Minimal Python bridge for PODEM stuck-at fault catalogs";
  module.def("catalog_stuck_at", &catalog_stuck_at,
             py::arg("circuit_path"), py::arg("fault_map_path") = "");
  module.def("run_stuck_at_ordered", &run_stuck_at_ordered,
             py::arg("circuit_path"), py::arg("fault_map_path"),
             py::arg("ordered_fault_ids"), py::arg("backtrack_limit") = 5000,
             py::arg("seed") = 14);
  py::class_<StuckAtSession>(module, "StuckAtSession")
      .def(py::init<const std::string &, const std::string &, int, int>(),
           py::arg("circuit_path"), py::arg("fault_map_path") = "",
           py::arg("backtrack_limit") = 5000, py::arg("seed") = 14)
      .def("catalog", &StuckAtSession::catalog)
      .def("remaining_fault_ids", &StuckAtSession::remaining_fault_ids)
      .def("step", &StuckAtSession::step, py::arg("fault_id"))
      .def("result", &StuckAtSession::result);
}
