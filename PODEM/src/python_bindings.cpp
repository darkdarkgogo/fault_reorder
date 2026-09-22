#include <chrono>
#include <cstdio>
#include <mutex>

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
  result["current_pattern_count"] = summary.current_pattern_count;
  result["finalized"] = summary.finalized;
  result["patterns_before_stc"] = summary.patterns_before_stc;
  result["patterns_after_stc"] = summary.patterns_after_stc;
  result["stc_removed_patterns"] = summary.stc_removed_patterns;
  result["stc_shuffle_attempts"] = summary.stc_shuffle_attempts;
  result["stc_coverage_preserved"] = summary.stc_coverage_preserved;
  result["detected_collapsed_faults"] = summary.detected_collapsed_faults;
  result["detected_equivalent_faults"] = summary.detected_equivalent_faults;
  result["uncollapsed_faults"] = summary.uncollapsed_faults;
  result["aborted_faults"] = summary.aborted_faults;
  result["redundant_faults"] = summary.redundant_faults;
  result["redundant_equivalent_faults"] = summary.redundant_equivalent_faults;
  result["podem_calls"] = summary.podem_calls;
  result["total_backtracks"] = summary.total_backtracks;
  result["primary_podem_calls"] = summary.primary_podem_calls;
  result["dtc_secondary_calls"] = summary.dtc_secondary_calls;
  result["primary_backtracks"] = summary.primary_backtracks;
  result["dtc_backtracks"] = summary.dtc_backtracks;
  return result;
}

py::dict protocol_to_dict(const StuckAtProtocolConfig &config) {
  py::dict result;
  result["primary_backtrack_limit"] = config.primary_backtrack_limit;
  result["primary_seed"] = config.primary_seed;
  result["attempts_per_primary_fault"] = config.attempts_per_primary_fault;
  result["dtc_enabled"] = config.dtc_enabled;
  result["dtc_secondary_backtrack_limit"] =
      config.dtc_secondary_backtrack_limit;
  result["stc_enabled"] = config.stc_enabled;
  result["stc_reverse_order_enabled"] = config.stc_reverse_order_enabled;
  result["stc_shuffle_seed"] = config.stc_shuffle_seed;
  result["stc_no_improvement_limit"] = config.stc_no_improvement_limit;
  result["scoap_enabled"] = config.scoap_enabled;
  return result;
}

StuckAtProtocolConfig make_protocol_config(
    int backtrack_limit, int seed, bool dtc_enabled, bool stc_enabled) {
  // Explicit compression overrides support regression tests only. Production
  // callers use the enabled defaults of the fixed protocol.
  StuckAtProtocolConfig config;
  config.primary_backtrack_limit = backtrack_limit;
  config.primary_seed = seed;
  config.dtc_enabled = dtc_enabled;
  config.stc_enabled = stc_enabled;
  return config;
}

py::dict catalog_stuck_at(const std::string &circuit_path,
                          const std::string &fault_map_path) {
  ATPG atpg;
  using Clock = std::chrono::steady_clock;
  const Clock::time_point catalog_started = Clock::now();
  atpg.detected_num = 1;
  atpg.set_fault_map_path(fault_map_path);

  std::vector<ATPG::FaultCatalogEntry> catalog;
  int uncollapsed_total = 0;

  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    std::fflush(stdout);
    std::fprintf(
        stderr,
        "[LOAD][CATALOG] fault_list done circuit=%s elapsed_s=%.3f\n",
        circuit_path.c_str(),
        std::chrono::duration<double>(Clock::now() - catalog_started).count());
    std::fflush(stderr);

    const Clock::time_point copy_started = Clock::now();
    std::fprintf(stderr, "[LOAD][CATALOG] get_fault_catalog start circuit=%s\n",
                 circuit_path.c_str());
    std::fflush(stderr);
    catalog = atpg.get_fault_catalog();
    uncollapsed_total = atpg.get_uncollapsed_fault_count();
    std::fprintf(
        stderr,
        "[LOAD][CATALOG] get_fault_catalog done circuit=%s faults=%zu elapsed_s=%.3f\n",
        circuit_path.c_str(), catalog.size(),
        std::chrono::duration<double>(Clock::now() - copy_started).count());
    std::fflush(stderr);
  }

  const Clock::time_point dict_started = Clock::now();
  std::fprintf(stderr, "[LOAD][CATALOG] catalog_to_dict start circuit=%s faults=%zu\n",
               circuit_path.c_str(), catalog.size());
  std::fflush(stderr);
  py::dict result = catalog_to_dict(catalog, uncollapsed_total);
  std::fprintf(
      stderr,
      "[LOAD][CATALOG] catalog_to_dict done circuit=%s faults=%zu elapsed_s=%.3f\n",
      circuit_path.c_str(), catalog.size(),
      std::chrono::duration<double>(Clock::now() - dict_started).count());
  std::fflush(stderr);
  return result;
}

py::dict run_stuck_at_ordered(const std::string &circuit_path,
                              const std::string &fault_map_path,
                              const std::vector<std::string> &ordered_fault_ids,
                              int backtrack_limit, int seed,
                              bool dtc_enabled, bool stc_enabled) {
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_fault_map_path(fault_map_path);
  atpg.configure_ordered_stuck_at(make_protocol_config(
      backtrack_limit, seed, dtc_enabled, stc_enabled));

  ATPG::AtpgRunResult summary;
  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    std::fflush(stdout);
    atpg.reorder_stuck_at_faults(ordered_fault_ids);
    summary = atpg.run_stuck_at(false);
  }

  return summary_to_dict(summary);
}

class StuckAtSession {
public:
  StuckAtSession(const std::string &circuit_path,
                 const std::string &fault_map_path,
                 int backtrack_limit, int seed,
                 bool dtc_enabled, bool stc_enabled)
      : config_(make_protocol_config(
            backtrack_limit, seed, dtc_enabled, stc_enabled)) {
    atpg_.detected_num = 1;
    atpg_.set_fault_map_path(fault_map_path);
    atpg_.configure_ordered_stuck_at(config_);
    py::gil_scoped_release release;
    atpg_.input(circuit_path);
    atpg_.level_circuit();
    atpg_.rearrange_gate_inputs();
    atpg_.create_dummy_gate();
    atpg_.generate_fault_list();
    std::fflush(stdout);
    catalog_ = atpg_.get_fault_catalog();
    uncollapsed_total_ = atpg_.get_uncollapsed_fault_count();
  }

  py::dict catalog() const {
    return catalog_to_dict(catalog_, uncollapsed_total_);
  }

  std::vector<std::string> remaining_fault_ids() const {
    std::vector<std::string> result;
    {
      // Release the GIL before waiting for the instance lock, and unlock before
      // reacquiring it. The same ordering is required by step() and result().
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      result = atpg_.get_selectable_fault_ids();
    }
    return result;
  }

  py::dict step(
      const std::string &fault_id,
      const std::vector<std::string> &ranked_secondary_fault_ids) {
    ATPG::AtpgStepResult step_result;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      step_result = atpg_.step_stuck_at(
          fault_id, ranked_secondary_fault_ids);
    }
    py::dict result = summary_to_dict(step_result.cumulative_result);
    result["selected_fault_id"] = step_result.selected_fault_id;
    result["target_status"] = step_result.target_status;
    result["generated_pattern"] = step_result.generated_pattern;
    result["generated_test_vector"] = step_result.generated_test_vector;
    result["dtc_attempted_fault_ids"] = step_result.dtc_attempted_fault_ids;
    result["dtc_embedded_fault_ids"] = step_result.dtc_embedded_fault_ids;
    result["current_dtc_secondary_calls"] = step_result.current_dtc_secondary_calls;
    result["current_primary_backtracks"] = step_result.current_primary_backtracks;
    result["current_dtc_backtracks"] = step_result.current_dtc_backtracks;
    result["newly_detected_fault_ids"] = step_result.newly_detected_fault_ids;
    result["remaining_fault_ids"] = step_result.remaining_fault_ids;
    result["current_pattern_count"] =
        step_result.cumulative_result.current_pattern_count;
    result["current_podem_calls"] =
        step_result.cumulative_result.podem_calls;
    result["current_total_backtracks"] =
        step_result.cumulative_result.total_backtracks;
    return result;
  }

  py::dict step_legacy(const std::string &fault_id) {
    std::vector<std::string> secondaries;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      const std::vector<std::string> remaining =
          atpg_.get_selectable_fault_ids();
      for (const std::string &identifier : remaining)
        if (identifier != fault_id)
          secondaries.push_back(identifier);
    }
    return step(fault_id, secondaries);
  }

  py::dict result() {
    ATPG::AtpgRunResult summary;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      summary = atpg_.finalize_stuck_at_session();
    }
    return summary_to_dict(summary);
  }

  py::dict config() const {
    return protocol_to_dict(config_);
  }

private:
  ATPG atpg_;
  const StuckAtProtocolConfig config_;
  std::vector<ATPG::FaultCatalogEntry> catalog_;
  int uncollapsed_total_{};
  mutable std::mutex mutex_;
};

} // namespace

PYBIND11_MODULE(cpp_podem, module) {
  module.doc() = "Minimal Python bridge for PODEM stuck-at fault catalogs";
  module.def("catalog_stuck_at", &catalog_stuck_at,
             py::arg("circuit_path"), py::arg("fault_map_path") = "");
  module.def("run_stuck_at_ordered", &run_stuck_at_ordered,
             py::arg("circuit_path"), py::arg("fault_map_path"),
             py::arg("ordered_fault_ids"), py::arg("backtrack_limit") = 100,
             py::arg("seed") = 14, py::arg("dtc_enabled") = true,
             py::arg("stc_enabled") = true);
  py::class_<StuckAtSession>(module, "StuckAtSession")
      .def(py::init<const std::string &, const std::string &, int, int,
                    bool, bool>(),
           py::arg("circuit_path"), py::arg("fault_map_path") = "",
           py::arg("backtrack_limit") = 100, py::arg("seed") = 14,
           py::arg("dtc_enabled") = true, py::arg("stc_enabled") = true)
      .def("catalog", &StuckAtSession::catalog)
      .def("config", &StuckAtSession::config)
      .def("remaining_fault_ids", &StuckAtSession::remaining_fault_ids)
      .def("step", &StuckAtSession::step,
           py::arg("fault_id"), py::arg("ranked_secondary_fault_ids"))
      .def("step", &StuckAtSession::step_legacy, py::arg("fault_id"))
      .def("result", &StuckAtSession::result);
}
