/**********************************************************************/
/*           Constructors of the classes defined in atpg.h            */
/*           ATPG top-level functions                                 */
/*           Author: Bing-Chen (Benson) Wu                            */
/*           last update : 01/21/2018                                 */
/**********************************************************************/

#include "atpg.h"

#include <chrono>
#include <cstdio>

namespace
{
	double steady_seconds()
	{
		return std::chrono::duration<double>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	bool atpg_timing_enabled()
	{
#ifdef _MSC_VER
		char value[2]{};
		size_t required = 0;
		getenv_s(&required, value, sizeof(value), "PODEM_ATPG_TIMING");
		return required == 2 && value[0] == '1';
#else
		const char *value = std::getenv("PODEM_ATPG_TIMING");
		return value != nullptr && std::strcmp(value, "1") == 0;
#endif
	}
}
void ATPG::configure_ordered_stuck_at(const StuckAtProtocolConfig &config)
{
	stuck_at_protocol_config = config;
	SAF_atpg = true;
	fsim_only = false;
	tdfsim_only = false;
	total_attempt_num = config.attempts_per_primary_fault;
	backtrack_limit = config.primary_backtrack_limit;
	seed = config.primary_seed;
	fault_order_by_scoap = config.scoap_enabled;
	dynamic_test_compression = config.dtc_enabled;
	static_test_compression = config.stc_enabled;
	podemx_backtrack_limit = config.dtc_secondary_backtrack_limit;
	stcseed = config.stc_shuffle_seed;
	stctime = -config.stc_no_improvement_limit;
	print_test_vectors = false;
	primary_fill_rng.seed(config.primary_seed);
	stc_shuffle_rng.seed(config.stc_shuffle_seed);
	stuck_at_session_prepared = false;
	stuck_at_final_result = AtpgRunResult{};
	stuck_at_total_detect_num = 0;
	stuck_at_detected_collapsed_faults = 0;
	stuck_at_total_backtracks = 0;
	stuck_at_aborted_faults = 0;
	stuck_at_redundant_faults = 0;
	stuck_at_redundant_equivalent_faults = 0;
	stuck_at_podem_calls = 0;
	stuck_at_dtc_secondary_calls = 0;
	stuck_at_primary_backtracks = 0;
	stuck_at_dtc_backtracks = 0;
	reset_stuck_at_active_step();
}

void ATPG::prepare_stuck_at_session()
{
	if (!SAF_atpg || fsim_only || tdfsim_only || total_attempt_num != 1)
		throw runtime_error("Ordered ATPG requires one-attempt stuck-at generation mode");
	if (stuck_at_session_prepared)
		return;
	stuck_at_faults_by_id.clear();
	for (const auto &owned_fault : flist)
	{
		fptr fault = owned_fault.get();
		const string identifier = fault_identifier(fault);
		if (!stuck_at_faults_by_id.emplace(identifier, fault).second)
			throw runtime_error("Duplicate fault ID in catalog: " + identifier);
	}

	// Backtrace supports AND/OR/NAND/NOR/NOT/BUF, not physical XOR/EQV.
	// Reject these before a sampled order can reach an undefined backtrace path.
	for (wptr wire : sort_wlist)
	{
		if (!wire->inode.empty() &&
			(wire->inode.front()->type == XOR || wire->inode.front()->type == EQV))
			throw runtime_error("Stuck-at PODEM requires physical XOR/EQV gates to be expanded before ordering");
	}

	// Also honor seeds supplied by the legacy CLI, which does not call
	// configure_ordered_stuck_at(). No stuck-at path uses the C global RNG.
	primary_fill_rng.seed(seed);
	stc_shuffle_rng.seed(stcseed);
	for (wptr wire : sort_wlist)
		wire->udflist.clear();
	vector<vector<fptr>> faults_by_wire(sort_wlist.size());
	for (fptr fault : flist_undetect)
		if (fault->to_swlist >= 0 &&
			static_cast<size_t>(fault->to_swlist) < faults_by_wire.size())
			faults_by_wire[fault->to_swlist].push_back(fault);
	for (size_t index = 0; index < faults_by_wire.size(); ++index)
		for (auto fault = faults_by_wire[index].rbegin();
			fault != faults_by_wire[index].rend(); ++fault)
			sort_wlist[index]->udflist.push_front(*fault);
	stuck_at_session_prepared = true;
}

void ATPG::fill_stuck_at_primary_cube()
{
	uniform_int_distribution<int> bit(0, 1);
	for (wptr wire : cktin)
	{
		switch (wire->value)
		{
			case D:
				wire->value = 1;
				break;
			case D_bar:
				wire->value = 0;
				break;
			case U:
				wire->value = bit(primary_fill_rng);
				break;
			default:
				break;
		}
	}
}

vector<string> ATPG::get_selectable_fault_ids() const
{
	vector<string> result;
	for (fptr fault : flist_undetect)
	{
		if (!fault->test_tried && fault->detect != REDUNDANT)
			result.push_back(fault_identifier(fault));
	}
	return result;
}

ATPG::AtpgRunResult ATPG::get_stuck_at_result() const
{
	if (stuck_at_final_result.finalized)
		return stuck_at_final_result;
	AtpgRunResult result;
	result.pattern_count = in_vector_no;
	result.current_pattern_count = in_vector_no;
	result.patterns_before_stc = in_vector_no;
	result.patterns_after_stc = in_vector_no;
	result.detected_equivalent_faults = stuck_at_total_detect_num;
	result.uncollapsed_faults = num_of_gate_fault;
	result.aborted_faults = stuck_at_aborted_faults;
	result.redundant_faults = stuck_at_redundant_faults;
	result.redundant_equivalent_faults = stuck_at_redundant_equivalent_faults;
	result.podem_calls = stuck_at_podem_calls;
	result.total_backtracks = stuck_at_total_backtracks;
	result.primary_podem_calls = stuck_at_podem_calls;
	result.dtc_secondary_calls = stuck_at_dtc_secondary_calls;
	result.primary_backtracks = stuck_at_primary_backtracks;
	result.dtc_backtracks = stuck_at_dtc_backtracks;
	result.detected_collapsed_faults = stuck_at_detected_collapsed_faults;
	return result;
}

void ATPG::reset_stuck_at_active_step()
{
	stuck_at_step_phase = StuckAtStepPhase::idle;
	stuck_at_active_primary = nullptr;
	stuck_at_active_unknown_po = nullptr;
	stuck_at_active_candidates.clear();
	stuck_at_accepted_pi_cube.clear();
	stuck_at_attempted_ids.clear();
	stuck_at_active_step = AtpgStepResult{};
	stuck_at_active_dtc_calls = 0;
	stuck_at_active_dtc_backtracks = 0;
	stuck_at_active_primary_backtracks = 0;
	stuck_at_active_batch_index = 0;
	stuck_at_active_select_fault_try = 0;
	stuck_at_active_visited_wire_count = 0;
	stuck_at_next_po_index = 0;
	stuck_at_active_timing_enabled = false;
	stuck_at_active_step_started_seconds = 0.0;
	stuck_at_active_book_elapsed_seconds = 0.0;
}

ATPG::StuckAtPhaseResult ATPG::begin_stuck_at_step(const string &fault_id)
{
	return begin_stuck_at_step_impl(fault_id, true);
}

ATPG::StuckAtPhaseResult ATPG::begin_stuck_at_step_impl(
	const string &fault_id, bool expose_ranked_dtc)
{
	prepare_stuck_at_session();
	if (stuck_at_step_phase != StuckAtStepPhase::idle)
		throw runtime_error("A stuck-at step is already awaiting a DTC ranking");
	const bool timing_enabled = atpg_timing_enabled();
	const double step_started_seconds =
		timing_enabled ? steady_seconds() : 0.0;
	const auto known = stuck_at_faults_by_id.find(fault_id);
	if (known == stuck_at_faults_by_id.end())
		throw runtime_error("Unknown fault ID: " + fault_id);
	fptr fault_under_test = known->second;
	if (fault_under_test->detect != FALSE || fault_under_test->test_tried)
		throw runtime_error("Fault ID is not selectable: " + fault_id);

	size_t selectable_count = 0;
	for (fptr fault : flist_undetect)
		if (!fault->test_tried && fault->detect != REDUNDANT)
			++selectable_count;
	const double begin_book_elapsed_seconds = timing_enabled
		? steady_seconds() - step_started_seconds
		: 0.0;

	reset_stuck_at_active_step();
	stuck_at_active_timing_enabled = timing_enabled;
	stuck_at_active_step_started_seconds = step_started_seconds;
	stuck_at_active_book_elapsed_seconds = begin_book_elapsed_seconds;
	stuck_at_step_phase = StuckAtStepPhase::awaiting_dtc_order;
	stuck_at_active_primary = fault_under_test;
	stuck_at_active_step.selected_fault_id = fault_id;
	int current_backtracks = 0;
	using Clock = std::chrono::steady_clock;
	const Clock::time_point primary_started = Clock::now();
	fprintf(
		stderr,
		"[ATPG][PRIMARY] start fault=%s selectable=%zu backtrack_limit=%d\n",
		fault_id.c_str(), selectable_count, backtrack_limit);
	fflush(stderr);
	const int podem_result = podem(fault_under_test, current_backtracks);
	const char *primary_status =
		podem_result == TRUE ? "detected" :
		podem_result == FALSE ? "redundant" :
		podem_result == MAYBE ? "aborted" : "unsupported";
	fprintf(
		stderr,
		"[ATPG][PRIMARY] done fault=%s status=%s backtracks=%d elapsed_s=%.3f\n",
		fault_id.c_str(), primary_status, current_backtracks,
		std::chrono::duration<double>(Clock::now() - primary_started).count());
	fflush(stderr);
	stuck_at_active_primary_backtracks = current_backtracks;
	stuck_at_total_backtracks += current_backtracks;
	stuck_at_primary_backtracks += current_backtracks;
	stuck_at_podem_calls++;
	switch (podem_result)
	{
		case TRUE:
		{
			stuck_at_active_step.target_status = "detected";
			stuck_at_active_step.generated_pattern = true;
			for (wptr wire : cktin)
				stuck_at_accepted_pi_cube.push_back(
					wire->value == D ? 1 : wire->value == D_bar ? 0 : wire->value);
			restore_stuck_at_good_cube(stuck_at_accepted_pi_cube);
			if (dynamic_test_compression)
			{
				if (expose_ranked_dtc)
				{
					DtcBatchState batch;
					if (find_next_stuck_at_dtc_batch(batch))
					{
						stuck_at_active_unknown_po = batch.unknown_po;
						stuck_at_active_candidates = batch.candidates;
						stuck_at_active_select_fault_try = batch.select_fault_try;
						stuck_at_active_visited_wire_count = batch.visited_wire_count;
						return make_stuck_at_dtc_phase();
					}
				}
				else
					run_stuck_at_lazy_dtc();
			}
			break;
		}
		case FALSE:
			fault_under_test->detect = REDUNDANT;
			stuck_at_redundant_faults++;
			stuck_at_redundant_equivalent_faults += fault_under_test->eqv_fault_num;
			stuck_at_active_step.target_status = "redundant";
			break;
		case MAYBE:
			stuck_at_aborted_faults++;
			stuck_at_active_step.target_status = "aborted";
			break;
		default:
			reset_stuck_at_active_step();
			throw runtime_error("PODEM returned an unsupported status");
	}
	StuckAtPhaseResult result;
	result.phase = "complete";
	result.step_result = complete_stuck_at_step();
	return result;
}

ATPG::AtpgStepResult ATPG::complete_stuck_at_step()
{
	if (stuck_at_step_phase != StuckAtStepPhase::awaiting_dtc_order ||
		stuck_at_active_primary == nullptr)
		throw runtime_error("No active stuck-at step can be completed");
	double fsim_elapsed_seconds = 0.0;
	double book_elapsed_seconds = stuck_at_active_book_elapsed_seconds;
	int current_detect_num = 0;
	if (stuck_at_active_step.generated_pattern)
	{
		restore_stuck_at_good_cube(stuck_at_accepted_pi_cube);
		fill_stuck_at_primary_cube();
		string vec;
		for (wptr wire : cktin)
			vec.push_back(itoc(wire->value));
		stuck_at_active_step.generated_test_vector = vec;
		if (print_test_vectors)
			display_io();
		vector<fptr> newly_detected_faults;
		const double fsim_started_seconds = stuck_at_active_timing_enabled
			? steady_seconds()
			: 0.0;
		fault_sim_a_vector(vec, current_detect_num, &newly_detected_faults);
		if (stuck_at_active_timing_enabled)
			fsim_elapsed_seconds = steady_seconds() - fsim_started_seconds;
		const double detection_book_started_seconds =
			stuck_at_active_timing_enabled ? steady_seconds() : 0.0;
		for (fptr fault : newly_detected_faults)
			stuck_at_active_step.newly_detected_fault_ids.push_back(
				fault_identifier(fault));
		stuck_at_detected_collapsed_faults +=
			static_cast<int>(newly_detected_faults.size());
		if (stuck_at_active_timing_enabled)
			book_elapsed_seconds +=
				steady_seconds() - detection_book_started_seconds;
		vectors.push_back(vec);
		stuck_at_total_detect_num += current_detect_num;
		in_vector_no++;
	}
	const double result_book_started_seconds = stuck_at_active_timing_enabled
		? steady_seconds()
		: 0.0;
	stuck_at_active_primary->test_tried = true;
	stuck_at_dtc_secondary_calls += stuck_at_active_dtc_calls;
	stuck_at_dtc_backtracks += stuck_at_active_dtc_backtracks;
	stuck_at_total_backtracks += stuck_at_active_dtc_backtracks;
	stuck_at_active_step.current_dtc_secondary_calls = stuck_at_dtc_secondary_calls;
	stuck_at_active_step.current_primary_backtracks = stuck_at_primary_backtracks;
	stuck_at_active_step.current_dtc_backtracks = stuck_at_dtc_backtracks;

	stuck_at_active_step.remaining_fault_ids = get_selectable_fault_ids();
	stuck_at_active_step.cumulative_result = get_stuck_at_result();
	AtpgStepResult result = stuck_at_active_step;
	double step_elapsed_seconds = 0.0;
	if (stuck_at_active_timing_enabled)
	{
		book_elapsed_seconds += steady_seconds() - result_book_started_seconds;
		step_elapsed_seconds =
			steady_seconds() - stuck_at_active_step_started_seconds;
		fprintf(
			stderr,
			"[ATPG][FSIM] detected_equivalent=%d elapsed_s=%.6f\n",
			current_detect_num, fsim_elapsed_seconds);
		fprintf(
			stderr,
			"[ATPG][BOOK] remaining=%zu newly_detected=%zu elapsed_s=%.6f\n",
			result.remaining_fault_ids.size(),
			result.newly_detected_fault_ids.size(), book_elapsed_seconds);
		fprintf(
			stderr,
			"[ATPG][STEP] scope=wall_including_ranker fault=%s status=%s elapsed_s=%.6f\n",
			result.selected_fault_id.c_str(), result.target_status.c_str(),
			step_elapsed_seconds);
		fflush(stderr);
	}
	reset_stuck_at_active_step();
	return result;
}

ATPG::AtpgStepResult ATPG::step_stuck_at(const string &fault_id)
{
	StuckAtPhaseResult state = begin_stuck_at_step_impl(fault_id, false);
	if (state.phase != "complete")
		throw runtime_error("Stuck-at step ended in an unsupported phase");
	return state.step_result;
}

ATPG::AtpgRunResult ATPG::run_stuck_at(bool print_report)
{
	prepare_stuck_at_session();
	vector<string> selectable = get_selectable_fault_ids();
	while (!selectable.empty())
		selectable = step_stuck_at(selectable.front()).remaining_fault_ids;

	const AtpgRunResult result = finalize_stuck_at_session();
	if (print_report)
	{
		display_undetect();
		fprintf(stdout, "\n#number of aborted faults = %d\n", result.aborted_faults);
		fprintf(stdout, "\n#number of redundant faults = %d\n", result.redundant_faults);
		fprintf(stdout, "\n#number of calling podem1 = %d\n", result.podem_calls);
		fprintf(stdout, "\n#total number of backtracks = %d\n", result.total_backtracks);
	}
	return result;
}

void ATPG::test()
{
	string vec;
	int current_detect_num = 0;
	int total_detect_num = 0;
	int total_no_of_backtracks = 0; // accumulative number of backtracks
	int current_backtracks = 0;
	int no_of_aborted_faults = 0;
	int no_of_redundant_faults = 0;
	int no_of_calls = 0;

	fptr fault_under_test = flist_undetect.front();

	/* stuck-at fault sim mode */
	if (fsim_only)
	{
		fault_simulate_vectors(total_detect_num);
		in_vector_no += vectors.size();
		display_undetect();
		fprintf(stdout, "\n");
		return;
	} // if fsim only

	/* transition fault sim mode */
	if (tdfsim_only)
	{
		transition_delay_fault_simulation(total_detect_num);
		in_vector_no += vectors.size();
		display_undetect();

		printf("\n# Result:\n");
		printf("-----------------------\n");
		printf("# total transition delay faults: %d\n", num_of_tdf_fault);
		printf("# total detected faults: %d\n", total_detect_num);
		printf("# fault coverage: %lf %\n", (double)total_detect_num / (double)num_of_tdf_fault * 100);
		return;
	} // if fsim only

	/* SAF test generation mode */
	if (SAF_atpg)
	{
		run_stuck_at(true);
		return;
	}

	/* TDF test generation mode */
	int cur_seed = 0;
	srand(seed == -1 ? cur_seed : seed);
	cur_i = 1;
	ncktwire = sort_wlist.size();
	ncktin = cktin.size();
	if (flow == 1)
	{
		while (cur_i <= detected_num)
		{
			if (cur_i > 1)
			{
				for (fptr fptr_ele : flist_undetect)
				{
					if (!fptr_ele->test_tried && (fptr_ele->detected_time == cur_i - 1))
					{
						fault_under_test = fptr_ele;
						break;
					}
				}
			}
			while (fault_under_test != nullptr)
			{

				switch (tdf_podem(fault_under_test, current_backtracks))
				{
					case TRUE:
						if (seed == -1)
						{
							srand(cur_seed++);
						}
						vec.clear();
						for (int i = 0; i < cktin.size(); i++)
						{
							if (cktin[i]->value == U)
							{
								cktin[i]->value = rand() & 01;
							}
						}
						if (last_bit == U)
						{
							last_bit = rand() & 01;
						}
						for (int i = 1; i < cktin.size(); i++)
						{
							vec.push_back(itoc(cktin[i]->value));
						}
						vec.push_back(itoc(last_bit));
						vec.push_back(itoc(cktin[0]->value));
						vectors.push_back(vec);
						
						tdfault_sim_a_vector(vec, current_detect_num);
						total_detect_num += current_detect_num;

						break;
					case FALSE:
						fault_under_test->detect = REDUNDANT;
						no_of_redundant_faults++;
						fault_under_test->test_tried = true;
						break;

					case MAYBE:
						no_of_aborted_faults++;
						fault_under_test->test_tried = true;
						break;
				}

				fault_under_test = nullptr;
				for (fptr fptr_ele : flist_undetect)
				{
					if (!fptr_ele->test_tried && (fptr_ele->detected_time == cur_i - 1))
					{
						fault_under_test = fptr_ele;
						break;
					}
				}
				total_no_of_backtracks += current_backtracks; // accumulate number of backtracks
				no_of_calls++;
			}
			cur_i++;
		}
	}
	else
	{
		while (fault_under_test != nullptr)
		{
			switch (tdf_podem(fault_under_test, current_backtracks))
			{
				case TRUE:
					/* form a vector */
					if (seed == -1)
					{
						srand(cur_seed++);
					}
					vec.clear();
					for (int i = 0; i < cktin.size(); i++)
					{
						if (cktin[i]->value == U)
						{
							cktin[i]->value = rand() & 01;
						}
					}
					if (last_bit == U)
					{
						last_bit = rand() & 01;
					}
					for (int i = 1; i < cktin.size(); i++)
					{
						vec.push_back(itoc(cktin[i]->value));
					}
					vec.push_back(itoc(last_bit));
					vec.push_back(itoc(cktin[0]->value));
					vectors.push_back(vec);

					tdfault_sim_a_vector(vec, current_detect_num);
					total_detect_num += current_detect_num;
					break;
				case FALSE:
					fault_under_test->detect = REDUNDANT;
					no_of_redundant_faults++;
					fault_under_test->test_tried = true;
					break;

				case MAYBE:
					no_of_aborted_faults++;
					fault_under_test->test_tried = true;
					break;
			}
			fault_under_test = nullptr;
			for (fptr fptr_ele : flist_undetect)
			{
				if (!fptr_ele->test_tried)
				{
					fault_under_test = fptr_ele;
					break;
				}
			}
			total_no_of_backtracks += current_backtracks; // accumulate number of backtracks
			no_of_calls++;
		}
	}
	if (static_test_compression)
	{
		reverse_order_fault_sim();
		if (stctime >= 0)
		{
			while (stctime--)
			{
				random_order_fault_sim();
			}
		}
		else
		{
			stctime *= (-1);
			int fail = 0;
			int v_size = vectors.size();
			while (fail < stctime)
			{
				random_order_fault_sim();
				if (vectors.size() == v_size)
				{
					fail++;
				}
				else
				{
					v_size = vectors.size();
					fail = 0;
				}
			}
		}
	}

	in_vector_no = vectors.size();
	fprintf(stdout, "\n");
	for (int i = 0; i < vectors.size(); i++)
	{
		fprintf(stdout, "T\'");
		fprintf(stdout, vectors[i].substr(0, vectors[i].size() - 1).c_str());
		fprintf(stdout, " ");
		fprintf(stdout, &vectors[i].back());
		fprintf(stdout, "\'\n");
	}
	display_undetect();
	fprintf(stdout, "\n");
	fprintf(stdout, "#number of aborted faults = %d\n", no_of_aborted_faults);
	fprintf(stdout, "\n");
	fprintf(stdout, "#number of redundant faults = %d\n", no_of_redundant_faults);
	fprintf(stdout, "\n");
	fprintf(stdout, "#number of calling podem1 = %d\n", no_of_calls);
	fprintf(stdout, "\n");
	fprintf(stdout, "#total number of backtracks = %d\n", total_no_of_backtracks);
} /* end of test */

/* constructor of ATPG */
ATPG::ATPG()
{
	/* orginally assigned in tpgmain.c */
	this->backtrack_limit = 3000; /* default value */
	this->total_attempt_num = 1; /* default value */
	this->fsim_only = false;		 /* flag to indicate fault simulation only */
	this->tdfsim_only = false;	 /* flag to indicate tdfault simulation only */

	/* orginally assigned in input.c */
	this->debug = 0;	 /* != 0 if debugging;  this is a switch of debug mode */
	this->lineno = 0;	 /* current line number */
	this->targc = 0;	 /* number of args on current command line */
	this->file_no = 0; /* number of current file */

	/* orginally assigned in init_flist.c */
	this->num_of_gate_fault = 0; // totle number of faults in the whole circuit

		/* orginally assigned in test.c */
	this->in_vector_no = 0; /* number of test vectors generated */
}

/* constructor of WIRE */
ATPG::WIRE::WIRE()
{
	this->value = 0;
	this->level = 0;
	this->wire_value1 = 0;
	this->wire_value2 = 0;
	this->wlist_index = 0;
}

/* constructor of NODE */
ATPG::NODE::NODE()
{
	this->type = 0;
	this->marked = false;
}

/* constructor of FAULT */
ATPG::FAULT::FAULT()
{
	this->node = nullptr;
	this->io = 0;
	this->index = 0;
	this->fault_type = 0;
	this->detect = 0;
	this->test_tried = false;
	this->eqv_fault_num = 0;
	this->to_swlist = 0;
	this->fault_no = 0;
	this->detected_time = 0;
	this->tried_dtc = false;
	this->logical_xor_input = false;
	this->logical_input_wire = nullptr;
	this->logical_input_index = -1;
	this->logical_input_occurrence = -1;
}
