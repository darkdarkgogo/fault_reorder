#include "atpg.h"

namespace {
int good_value(int value)
{
	return value == D ? 1 : (value == D_bar ? 0 : value);
}
}

// Rebuild the fault-free implication before injecting each fault. In particular,
// a D left on a PI by the preceding fault must not contaminate this evaluation.
// This is scalar cube implication, not fault simulation: no fault is dropped.
bool ATPG::stuck_at_cube_detects(fptr fault)
{
	for (wptr wire : sort_wlist)
	{
		wire->remove_changed();
		wire->remove_scheduled();
		if (wire->is_input())
			wire->value = good_value(wire->value);
		else
		{
			wire->value = U;
			evaluate(wire->inode.front());
			wire->remove_changed();
		}
	}
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
	for (wptr wire : cktin)
		wire->value = good_value(wire->value);
	return status;
}

ATPG::DtcResult ATPG::run_stuck_at_dtc(fptr primary)
{
	DtcResult result;
	if (!dynamic_test_compression)
		return result;

	vector<fptr> candidates;
	for (fptr fault : flist_undetect)
		if (fault != primary && !fault->test_tried && fault->detect != REDUNDANT)
			candidates.push_back(fault);
	// fault_no is assigned at catalog construction and survives primary reorders.
	sort(candidates.begin(), candidates.end(), [](fptr left, fptr right) {
		return left->fault_no < right->fault_no;
	});
	vector<fptr> preserved{primary};
	for (fptr secondary : candidates)
	{
		struct Snapshot { int value; bool assigned; bool assigned_v2; bool changed; bool scheduled; };
		vector<Snapshot> snapshot;
		for (wptr wire : sort_wlist)
			snapshot.push_back({wire->value, wire->is_all_assigned(),
				wire->is_all_assigned(true), wire->is_changed(), wire->is_scheduled()});
		result.attempted_fault_ids.push_back(fault_identifier(secondary));
		++result.secondary_calls;
		int backtracks = 0;
		bool embedded = stuck_at_podemx_secondary(secondary, backtracks) == TRUE;
		result.backtracks += backtracks;
		if (embedded)
		{
			for (fptr fault : preserved)
				if (!stuck_at_cube_detects(fault))
				{
					embedded = false;
					break;
				}
		}
		if (embedded)
		{
			preserved.push_back(secondary);
			result.embedded_fault_ids.push_back(fault_identifier(secondary));
			for (wptr wire : cktin)
				wire->value = good_value(wire->value);
		}
		else
		{
			// Restore both the PI cube and all scalar implication/decision flags;
			// neither failed nor limited secondary searches have lasting state.
			for (size_t i = 0; i < sort_wlist.size(); ++i)
			{
				wptr wire = sort_wlist[i];
				const Snapshot &saved = snapshot[i];
				wire->value = saved.value;
				wire->remove_all_assigned();
				wire->remove_all_assigned(true);
				wire->remove_changed();
				wire->remove_scheduled();
				if (saved.assigned) wire->set_all_assigned();
				if (saved.assigned_v2) wire->set_all_assigned(true);
				if (saved.changed) wire->set_changed();
				if (saved.scheduled) wire->set_scheduled();
			}
		}
	}
	return result;
}

ATPG::AtpgRunResult ATPG::finalize_stuck_at_session()
{
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
