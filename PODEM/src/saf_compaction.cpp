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
