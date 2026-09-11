/**********************************************************************/
/*           Constructors of the classes defined in atpg.h            */
/*           ATPG top-level functions                                 */
/*           Author: Bing-Chen (Benson) Wu                            */
/*           last update : 01/21/2018                                 */
/**********************************************************************/

#include "atpg.h"

void ATPG::configure_ordered_stuck_at(int configured_backtrack_limit,
		int configured_seed)
{
	SAF_atpg = true;
	fsim_only = false;
	tdfsim_only = false;
	total_attempt_num = 1;
	backtrack_limit = configured_backtrack_limit;
	seed = configured_seed;
	fault_order_by_scoap = false;
	dynamic_test_compression = false;
	static_test_compression = false;
	print_test_vectors = false;
}

ATPG::AtpgRunResult ATPG::run_stuck_at(bool print_report)
{
	if (!SAF_atpg || fsim_only || tdfsim_only || total_attempt_num != 1)
		throw runtime_error("Ordered ATPG requires one-attempt stuck-at generation mode");

	// Backtrace supports AND/OR/NAND/NOR/NOT/BUF, not physical XOR/EQV.
	// Reject these before a sampled order can reach an undefined backtrace path.
	for (wptr wire : sort_wlist)
	{
		if (!wire->inode.empty() &&
			(wire->inode.front()->type == XOR || wire->inode.front()->type == EQV))
			throw runtime_error("Stuck-at PODEM requires physical XOR/EQV gates to be expanded before ordering");
	}

	srand(seed);
	int current_detect_num = 0;
	int total_detect_num = 0;
	int total_no_of_backtracks = 0;
	int current_backtracks = 0;
	int no_of_aborted_faults = 0;
	int no_of_redundant_faults = 0;
	int no_of_redundant_equivalent_faults = 0;
	int no_of_calls = 0;
	fptr fault_under_test = flist_undetect.empty() ? nullptr : flist_undetect.front();

	while (fault_under_test != nullptr)
	{
		switch (podem(fault_under_test, current_backtracks))
		{
			case TRUE:
			{
				string vec;
				for (wptr w : cktin)
					vec.push_back(itoc(w->value));
				fault_sim_a_vector(vec, current_detect_num);
				total_detect_num += current_detect_num;
				in_vector_no++;
				break;
			}
			case FALSE:
				fault_under_test->detect = REDUNDANT;
				no_of_redundant_faults++;
				no_of_redundant_equivalent_faults += fault_under_test->eqv_fault_num;
				break;
			case MAYBE:
				no_of_aborted_faults++;
				break;
		}
		fault_under_test->test_tried = true;
		fault_under_test = nullptr;
		for (fptr candidate : flist_undetect)
		{
			if (!candidate->test_tried)
			{
				fault_under_test = candidate;
				break;
			}
		}
		total_no_of_backtracks += current_backtracks;
		no_of_calls++;
	}

	AtpgRunResult result;
	result.pattern_count = in_vector_no;
	result.detected_equivalent_faults = total_detect_num;
	result.uncollapsed_faults = num_of_gate_fault;
	result.aborted_faults = no_of_aborted_faults;
	result.redundant_faults = no_of_redundant_faults;
	result.redundant_equivalent_faults = no_of_redundant_equivalent_faults;
	result.podem_calls = no_of_calls;
	result.total_backtracks = total_no_of_backtracks;
	for (const auto &owned_fault : flist)
	{
		if (owned_fault->detect == TRUE)
			result.detected_collapsed_faults++;
	}

	if (print_report)
	{
		display_undetect();
		fprintf(stdout, "\n#number of aborted faults = %d\n", no_of_aborted_faults);
		fprintf(stdout, "\n#number of redundant faults = %d\n", no_of_redundant_faults);
		fprintf(stdout, "\n#number of calling podem1 = %d\n", no_of_calls);
		fprintf(stdout, "\n#total number of backtracks = %d\n", total_no_of_backtracks);
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

	int cur_seed = 0;
	if (seed == -1)
	{ // increase seed
		srand(cur_seed);
	}
	else
	{
		srand(seed);
	}

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
