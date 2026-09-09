#include <mutex>

#include <pybind11/pybind11.h>

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

} // namespace

PYBIND11_MODULE(cpp_podem, module) {
  module.doc() = "Minimal Python bridge for PODEM stuck-at fault catalogs";
  module.def("catalog_stuck_at", &catalog_stuck_at,
             py::arg("circuit_path"), py::arg("fault_map_path") = "");
}
