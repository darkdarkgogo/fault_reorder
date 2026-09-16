/**********************************************************************/
/*           Constructors of the classes defined in atpg.h            */
/*           ATPG top-level functions                                 */
/*           Author: Bing-Chen (Benson) Wu                            */
/*           last update : 01/21/2018                                 */
/**********************************************************************/

#include "atpg.h"

#include <unordered_set>

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
	stuck_at_total_detect_num = 0;
	stuck_at_total_backtracks = 0;
	stuck_at_aborted_faults = 0;
	stuck_at_redundant_faults = 0;
	stuck_at_redundant_equivalent_faults = 0;
	stuck_at_podem_calls = 0;
	stuck_at_dtc_secondary_calls = 0;
	stuck_at_primary_backtracks = 0;
	stuck_at_dtc_backtracks = 0;
}

void ATPG::prepare_stuck_at_session()
{
	if (!SAF_atpg || fsim_only || tdfsim_only || total_attempt_num != 1)
		throw runtime_error("Ordered ATPG requires one-attempt stuck-at generation mode");
	if (stuck_at_session_prepared)
		return;

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
	AtpgRunResult result;
	result.pattern_count = in_vector_no;
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
	for (const auto &owned_fault : flist)
	{
		if (owned_fault->detect == TRUE)
			result.detected_collapsed_faults++;
	}
	return result;
}

ATPG::AtpgStepResult ATPG::step_stuck_at(const string &fault_id)
{
	prepare_stuck_at_session();

	fptr known_fault = nullptr;
	for (const auto &owned_fault : flist)
	{
		if (fault_identifier(owned_fault.get()) == fault_id)
		{
			known_fault = owned_fault.get();
			break;
		}
	}
	if (known_fault == nullptr)
		throw runtime_error("Unknown fault ID: " + fault_id);

	fptr fault_under_test = nullptr;
	vector<string> before_ids;
	for (fptr fault : flist_undetect)
	{
		before_ids.push_back(fault_identifier(fault));
		if (fault == known_fault && !fault->test_tried && fault->detect != REDUNDANT)
			fault_under_test = fault;
	}
	if (fault_under_test == nullptr)
		throw runtime_error("Fault ID is not selectable: " + fault_id);

	AtpgStepResult step;
	step.selected_fault_id = fault_id;
	int current_backtracks = 0;
	const int podem_result = podem(fault_under_test, current_backtracks);
	switch (podem_result)
	{
		case TRUE:
		{
			const DtcResult dtc = run_stuck_at_dtc(fault_under_test);
			step.dtc_attempted_fault_ids = dtc.attempted_fault_ids;
			step.dtc_embedded_fault_ids = dtc.embedded_fault_ids;
			stuck_at_dtc_secondary_calls += dtc.secondary_calls;
			stuck_at_dtc_backtracks += dtc.backtracks;
			stuck_at_total_backtracks += dtc.backtracks;
			fill_stuck_at_primary_cube();
			string vec;
			for (wptr wire : cktin)
				vec.push_back(itoc(wire->value));
			step.generated_test_vector = vec;
			if (print_test_vectors)
				display_io();
			int current_detect_num = 0;
			fault_sim_a_vector(vec, current_detect_num);
			stuck_at_total_detect_num += current_detect_num;
			in_vector_no++;
			step.target_status = "detected";
			step.generated_pattern = true;
			break;
		}
		case FALSE:
			fault_under_test->detect = REDUNDANT;
			stuck_at_redundant_faults++;
			stuck_at_redundant_equivalent_faults += fault_under_test->eqv_fault_num;
			step.target_status = "redundant";
			break;
		case MAYBE:
			stuck_at_aborted_faults++;
			step.target_status = "aborted";
			break;
		default:
			throw runtime_error("PODEM returned an unsupported status");
	}
	fault_under_test->test_tried = true;
	stuck_at_total_backtracks += current_backtracks;
	stuck_at_primary_backtracks += current_backtracks;
	stuck_at_podem_calls++;
	step.current_dtc_secondary_calls = stuck_at_dtc_secondary_calls;
	step.current_primary_backtracks = stuck_at_primary_backtracks;
	step.current_dtc_backtracks = stuck_at_dtc_backtracks;

	const vector<string> undetected_ids = [&]() {
		vector<string> ids;
		for (fptr fault : flist_undetect)
			ids.push_back(fault_identifier(fault));
		return ids;
	}();
	const unordered_set<string> after_set(undetected_ids.begin(), undetected_ids.end());
	for (const string &identifier : before_ids)
	{
		if (after_set.find(identifier) == after_set.end())
			step.newly_detected_fault_ids.push_back(identifier);
	}
	step.remaining_fault_ids = get_selectable_fault_ids();
	step.cumulative_result = get_stuck_at_result();
	return step;
}

ATPG::AtpgRunResult ATPG::run_stuck_at(bool print_report)
{
	prepare_stuck_at_session();
	vector<string> selectable = get_selectable_fault_ids();
	while (!selectable.empty())
	{
		step_stuck_at(selectable.front());
		selectable = get_selectable_fault_ids();
	}

	const AtpgRunResult result = get_stuck_at_result();
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
