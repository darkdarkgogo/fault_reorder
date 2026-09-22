#include "atpg.h"

#include <chrono>
#include <cstdio>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace {
int good_value(int value)
{
	return value == D ? 1 : (value == D_bar ? 0 : value);
}
}

void ATPG::restore_stuck_at_good_cube(const vector<int> &accepted_pi_cube)
{
	if (accepted_pi_cube.size() != cktin.size())
		throw runtime_error("Accepted stuck-at PI cube has the wrong size");
	for (wptr wire : sort_wlist)
	{
		wire->remove_changed();
		wire->remove_scheduled();
		wire->remove_all_assigned();
		wire->remove_all_assigned(true);
	}
	for (size_t index = 0; index < cktin.size(); ++index)
	{
		cktin[index]->value = good_value(accepted_pi_cube[index]);
		cktin[index]->set_changed();
	}
	for (wptr wire : sort_wlist)
		if (!wire->is_input())
			wire->value = U;
	sim();
	for (wptr wire : sort_wlist)
	{
		wire->remove_changed();
		wire->remove_scheduled();
	}
}

// Rebuild the fault-free implication before injecting each fault. A D or
// D-bar left by the preceding injection must never contaminate this check.
bool ATPG::stuck_at_cube_detects(fptr fault)
{
	vector<int> cube;
	for (wptr wire : cktin)
		cube.push_back(good_value(wire->value));
	restore_stuck_at_good_cube(cube);
	if (wptr injected = fault_evaluate(fault))
		forward_imply(injected);
	return check_test();
}

// PODEMX uses the existing PODEM objective/backtrace on the current cube. Its
// private decision stack contains only PIs that were U at entry; fixed primary
// and earlier secondary assignments never become backtrack decisions.
int ATPG::stuck_at_podemx_secondary(fptr fault, int &backtracks)
{
	struct Decision { wptr wire; int first_value; bool flipped; };
	vector<Decision> decisions;
	vector<int> fixed_values;
	for (wptr wire : cktin)
		fixed_values.push_back(good_value(wire->value));
	backtracks = 0;
	mark_propagate_tree(fault->node);
	int status = FALSE;
	while (true)
	{
		if (stuck_at_cube_detects(fault))
		{
			status = TRUE;
			break;
		}
		vector<int> before;
		for (wptr wire : cktin)
			before.push_back(good_value(wire->value));
		wptr decision = test_possible(fault);
		bool valid_decision = decision != nullptr;
		for (size_t i = 0; i < cktin.size(); ++i)
		{
			// Guard the legacy backtrace's PI assignment boundary as well as
			// the decision stack: only a currently U PI may be assigned.
			if (cktin[i] == decision && before[i] != U)
				valid_decision = false;
			if (fixed_values[i] != U && good_value(cktin[i]->value) != fixed_values[i])
				valid_decision = false;
		}
		if (valid_decision)
		{
			decisions.push_back({decision, decision->value, false});
			continue;
		}
		for (size_t i = 0; i < cktin.size(); ++i)
			cktin[i]->value = before[i];
		while (!decisions.empty() && decisions.back().flipped)
		{
			decisions.back().wire->value = U;
			decisions.pop_back();
		}
		if (decisions.empty())
			break;
		if (backtracks >= podemx_backtrack_limit)
		{
			status = MAYBE;
			break;
		}
		Decision &last = decisions.back();
		last.wire->value = last.first_value ^ 1;
		last.flipped = true;
		++backtracks;
	}
	unmark_propagate_tree(fault->node);
	for (const Decision &decision : decisions)
		decision.wire->remove_all_assigned();
	for (wptr wire : cktin)
		wire->value = good_value(wire->value);
	return status;
}

bool ATPG::find_next_stuck_at_dtc_batch(DtcBatchState &batch)
{
	if (stuck_at_step_phase != StuckAtStepPhase::awaiting_dtc_order)
		throw runtime_error("Stuck-at DTC batch discovery requires an active step");
	const int budget = static_cast<int>(cktin.size()) <=
		stuck_at_protocol_config.dtc_bfs_small_input_threshold
		? stuck_at_protocol_config.dtc_bfs_small_select_fault_try
		: stuck_at_protocol_config.dtc_bfs_default_select_fault_try;
	for (; stuck_at_next_po_index < cktout.size(); ++stuck_at_next_po_index)
	{
		wptr unknown_po = cktout[stuck_at_next_po_index];
		if (unknown_po->value != U)
			continue;
		queue<wptr> pending;
		unordered_set<wptr> visited;
		unordered_set<string> candidate_ids;
		vector<fptr> candidates;
		pending.push(unknown_po);
		int visited_count = 0;
		while (!pending.empty() && visited_count < budget)
		{
			wptr wire = pending.front();
			pending.pop();
			if (!visited.insert(wire).second || wire->value != U)
				continue;
			++visited_count;
			for (fptr fault : wire->udflist)
			{
				const string identifier = fault_identifier(fault);
				if (fault == stuck_at_active_primary || fault->test_tried ||
					fault->detect == REDUNDANT ||
					stuck_at_attempted_ids.find(identifier) != stuck_at_attempted_ids.end())
					continue;
				if (candidate_ids.insert(identifier).second)
					candidates.push_back(fault);
			}
			if (!wire->inode.empty())
				for (wptr input : wire->inode.front()->iwire)
					if (input->value == U && visited.find(input) == visited.end())
						pending.push(input);
		}
		if (!candidates.empty())
		{
			batch.unknown_po = unknown_po;
			batch.candidates = candidates;
			batch.batch_index = stuck_at_active_batch_index++;
			batch.select_fault_try = budget;
			batch.visited_wire_count = visited_count;
			return true;
		}
	}
	return false;
}

ATPG::StuckAtPhaseResult ATPG::make_stuck_at_dtc_phase() const
{
	StuckAtPhaseResult result;
	result.phase = "dtc";
	result.selected_fault_id = stuck_at_active_step.selected_fault_id;
	result.unknown_po_id = stuck_at_active_unknown_po == nullptr
		? "" : stuck_at_active_unknown_po->name;
	for (fptr candidate : stuck_at_active_candidates)
		result.dtc_candidate_fault_ids.push_back(fault_identifier(candidate));
	result.dtc_batch_index = stuck_at_active_batch_index - 1;
	result.select_fault_try = stuck_at_active_select_fault_try;
	result.visited_wire_count = stuck_at_active_visited_wire_count;
	return result;
}

ATPG::StuckAtPhaseResult ATPG::rank_stuck_at_dtc_candidates(
	const vector<string> &ranked_candidate_fault_ids)
{
	if (stuck_at_step_phase != StuckAtStepPhase::awaiting_dtc_order ||
		stuck_at_active_unknown_po == nullptr)
		throw runtime_error("No stuck-at DTC batch is awaiting a ranking");
	unordered_map<string, fptr> expected;
	for (fptr candidate : stuck_at_active_candidates)
		expected.emplace(fault_identifier(candidate), candidate);
	if (ranked_candidate_fault_ids.size() != expected.size())
		throw runtime_error("Ranked DTC IDs must exactly cover the current BFS batch");
	vector<fptr> ranked;
	unordered_set<string> seen;
	for (const string &identifier : ranked_candidate_fault_ids)
	{
		auto found = expected.find(identifier);
		if (found == expected.end() || !seen.insert(identifier).second)
			throw runtime_error("Ranked DTC IDs must be a permutation of the current BFS batch");
		ranked.push_back(found->second);
	}

	using Clock = std::chrono::steady_clock;
	const Clock::time_point batch_started = Clock::now();
	const string primary_id = fault_identifier(stuck_at_active_primary);
	fprintf(stderr,
		"[ATPG][DTC] batch start primary=%s po=%s candidates=%zu budget=%d visited=%d\n",
		primary_id.c_str(), stuck_at_active_unknown_po->name.c_str(), ranked.size(),
		stuck_at_active_select_fault_try, stuck_at_active_visited_wire_count);
	fflush(stderr);
	const size_t attempted_before = stuck_at_active_step.dtc_attempted_fault_ids.size();
	const size_t embedded_before = stuck_at_active_step.dtc_embedded_fault_ids.size();
	for (fptr secondary : ranked)
	{
		const string identifier = fault_identifier(secondary);
		if (!stuck_at_attempted_ids.insert(identifier).second)
			throw runtime_error("A stuck-at DTC secondary was attempted more than once");
		stuck_at_active_step.dtc_attempted_fault_ids.push_back(identifier);
		++stuck_at_active_dtc_calls;
		int backtracks = 0;
		bool embedded = stuck_at_podemx_secondary(secondary, backtracks) == TRUE;
		stuck_at_active_dtc_backtracks += backtracks;
		vector<int> proposed_cube;
		for (wptr wire : cktin)
			proposed_cube.push_back(good_value(wire->value));
		if (embedded)
		{
			for (fptr preserved : stuck_at_preserved_faults)
			{
				restore_stuck_at_good_cube(proposed_cube);
				if (!stuck_at_cube_detects(preserved))
				{
					embedded = false;
					break;
				}
			}
		}
		if (embedded)
		{
			stuck_at_accepted_pi_cube = proposed_cube;
			stuck_at_preserved_faults.push_back(secondary);
			stuck_at_active_step.dtc_embedded_fault_ids.push_back(identifier);
		}
		restore_stuck_at_good_cube(stuck_at_accepted_pi_cube);
		if (stuck_at_active_unknown_po->value != U)
			break;
	}
	fprintf(stderr,
		"[ATPG][DTC] batch done primary=%s po=%s attempted=%zu embedded=%zu elapsed_s=%.3f\n",
		primary_id.c_str(), stuck_at_active_unknown_po->name.c_str(),
		stuck_at_active_step.dtc_attempted_fault_ids.size() - attempted_before,
		stuck_at_active_step.dtc_embedded_fault_ids.size() - embedded_before,
		std::chrono::duration<double>(Clock::now() - batch_started).count());
	fflush(stderr);

	++stuck_at_next_po_index;
	DtcBatchState next;
	if (find_next_stuck_at_dtc_batch(next))
	{
		stuck_at_active_unknown_po = next.unknown_po;
		stuck_at_active_candidates = next.candidates;
		stuck_at_active_select_fault_try = next.select_fault_try;
		stuck_at_active_visited_wire_count = next.visited_wire_count;
		return make_stuck_at_dtc_phase();
	}
	StuckAtPhaseResult result;
	result.phase = "complete";
	result.step_result = complete_stuck_at_step();
	return result;
}

ATPG::AtpgRunResult ATPG::finalize_stuck_at_session()
{
	if (stuck_at_step_phase != StuckAtStepPhase::idle)
		throw runtime_error("Cannot finalize while a stuck-at DTC batch is awaiting a ranking");
	if (stuck_at_final_result.finalized || !get_selectable_fault_ids().empty())
		return get_stuck_at_result();

	AtpgRunResult result = get_stuck_at_result();
	if (vectors.size() != static_cast<size_t>(in_vector_no))
		throw runtime_error("Stuck-at STC raw vector count mismatch");
	vector<string> compacted = vectors;
	mt19937 shuffle_rng = stc_shuffle_rng;
	vector<FAULT> detection_faults;
	vector<string> expected_ids;
	int expected_equivalents = 0;
	for (const auto &fault : flist)
	{
		if (fault->detect == TRUE)
		{
			// STC preserves the detected target set, not all catalog faults.
			// Replaying undetected/redundant entries can change legacy packet
			// flushing (a redundant tail skips the flush) and discover extra IDs.
			// Only clone targets already detected by the completed session.
			detection_faults.push_back(*fault);
			expected_ids.push_back(fault_identifier(fault.get()));
			expected_equivalents += fault->eqv_fault_num;
		}
	}
	if (expected_equivalents != result.detected_equivalent_faults)
		throw runtime_error("Stuck-at STC pre-compaction equivalent count mismatch");

	{
		// SAF simulation drops only these cloned FAULT records. Preserve every
		// wire field (including private decision/injection flags) and both lists
		// with an exception-safe scope guard. No live fault status or ATPG counter
		// is changed, even if coverage validation or an allocation throws.
		struct SimulationState
		{
			ATPG &owner;
			vector<WIRE> wires;
			forward_list<fptr> undetected;
			forward_list<wptr> faulty;
			explicit SimulationState(ATPG &atpg) : owner(atpg)
			{
				for (wptr wire : owner.sort_wlist)
					wires.push_back(*wire);
				undetected.swap(owner.flist_undetect);
				faulty.swap(owner.wlist_faulty);
				for (wptr wire : owner.sort_wlist)
				{
					wire->remove_changed();
					wire->remove_scheduled();
					wire->remove_faulty();
					wire->remove_fault_injected();
					wire->set_fault_free();
				}
			}
			~SimulationState()
			{
				for (size_t i = 0; i < wires.size(); ++i)
					swap(*owner.sort_wlist[i], wires[i]);
				owner.flist_undetect.swap(undetected);
				owner.wlist_faulty.swap(faulty);
			}
		} saved_state(*this);

		auto reset_detection = [&]() {
			flist_undetect.clear();
			for (auto it = detection_faults.rbegin(); it != detection_faults.rend(); ++it)
			{
				it->detect = FALSE;
				it->detected_time = 0;
				flist_undetect.push_front(&*it);
			}
		};
		auto validate_coverage = [&]() {
			vector<string> detected_ids;
			int equivalents = 0;
			for (FAULT &fault : detection_faults)
				if (fault.detect == TRUE)
				{
					detected_ids.push_back(fault_identifier(&fault));
					equivalents += fault.eqv_fault_num;
				}
			// Both lists follow the unchanged catalog order: vector equality is
			// exact ID-set equality, not merely equal coverage percentages.
			if (detected_ids != expected_ids || equivalents != expected_equivalents)
				throw runtime_error("Stuck-at STC changed detected catalog coverage");
		};
		auto verify_patterns = [&]() {
			reset_detection();
			for (const string &pattern : compacted)
			{
				int detected = 0;
				fault_sim_a_vector(pattern, detected);
			}
			validate_coverage();
		};
		auto compact_pass = [&](bool reverse_order) {
			reset_detection();
			vector<string> retained;
			for (size_t i = 0; i < compacted.size(); ++i)
			{
				const string &pattern = compacted[reverse_order ? compacted.size() - 1 - i : i];
				int detected = 0;
				fault_sim_a_vector(pattern, detected);
				if (detected > 0)
					retained.push_back(pattern);
			}
			validate_coverage();
			if (reverse_order)
				reverse(retained.begin(), retained.end());
			compacted.swap(retained);
		};

		verify_patterns();
		if (static_test_compression)
		{
			if (stuck_at_protocol_config.stc_reverse_order_enabled)
				compact_pass(true);
			int consecutive_failures = 0;
			while (consecutive_failures < stuck_at_protocol_config.stc_no_improvement_limit)
			{
				const size_t before = compacted.size();
				shuffle(compacted.begin(), compacted.end(), shuffle_rng);
				compact_pass(false);
				++result.stc_shuffle_attempts;
				consecutive_failures = compacted.size() < before ? 0 : consecutive_failures + 1;
			}
		}
		verify_patterns();
	}

	result.patterns_after_stc = static_cast<int>(compacted.size());
	result.pattern_count = result.patterns_after_stc;
	result.stc_removed_patterns = result.patterns_before_stc - result.patterns_after_stc;
	result.stc_coverage_preserved = true;
	result.finalized = true;
	// Commit only after all validation and state restoration. Keep in_vector_no
	// as the raw monotonic step counter. The binding holds the session mutex.
	vectors.swap(compacted);
	stc_shuffle_rng = shuffle_rng;
	stuck_at_final_result = result;
	return result;
}
