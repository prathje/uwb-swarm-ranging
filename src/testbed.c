#include "nodes.h"
// ----------------------
// Definition of Testbed lille
#if NUM_NODES>14
#error Testbed LILLE only has 14 nodes
#elif NUM_NODES<14
#warning Testbed LILLE has more nodes available (14 in total)
#endif
uint16_t node_ids[NUM_NODES] = {
	0x6b93, // dwm1001-1
	0xf1c3, // dwm1001-2
	0xc240, // dwm1001-3
	0x013f, // dwm1001-4
	0xb227, // dwm1001-5
	0x033b, // dwm1001-6
	0xf524, // dwm1001-7
	0x37b2, // dwm1001-8
	0x15ef, // dwm1001-9
	0x7e02, // dwm1001-10
	0xad36, // dwm1001-11
	0x3598, // dwm1001-12
	0x47e0, // dwm1001-13
	0x0e92, // dwm1001-14
};
int16_t node_factory_antenna_delay_offsets[NUM_NODES] = {
	8, // dwm1001-1
	9, // dwm1001-2
	7, // dwm1001-3
	22, // dwm1001-4
	11, // dwm1001-5
	12, // dwm1001-6
	7, // dwm1001-7
	22, // dwm1001-8
	22, // dwm1001-9
	22, // dwm1001-10
	-14, // dwm1001-11
	-9, // dwm1001-12
	9, // dwm1001-13
	-13, // dwm1001-14
};
float32_t node_distances[NUM_PAIRS] = {
	 1.8515, // dwm1001-2 to dwm1001-1
	 3.8663, // dwm1001-3 to dwm1001-1
	 2.4, // dwm1001-3 to dwm1001-2
	 4.8, // dwm1001-4 to dwm1001-1
	 3.8663, // dwm1001-4 to dwm1001-2
	 1.8515, // dwm1001-4 to dwm1001-3
	 2.7906, // dwm1001-5 to dwm1001-1
	 1.1697, // dwm1001-5 to dwm1001-2
	 2.0611, // dwm1001-5 to dwm1001-3
	 3.6807, // dwm1001-5 to dwm1001-4
	 4.7103, // dwm1001-6 to dwm1001-1
	 3.1636, // dwm1001-6 to dwm1001-2
	 1.1697, // dwm1001-6 to dwm1001-3
	 2.2152, // dwm1001-6 to dwm1001-4
	 2.4, // dwm1001-6 to dwm1001-5
	 4.1928, // dwm1001-7 to dwm1001-1
	 3.3407, // dwm1001-7 to dwm1001-2
	 3.747, // dwm1001-7 to dwm1001-3
	 4.8311, // dwm1001-7 to dwm1001-4
	 2.4, // dwm1001-7 to dwm1001-5
	 3.3941, // dwm1001-7 to dwm1001-6
	 5.655, // dwm1001-8 to dwm1001-1
	 4.4497, // dwm1001-8 to dwm1001-2
	 3.3407, // dwm1001-8 to dwm1001-3
	 3.834, // dwm1001-8 to dwm1001-4
	 3.3941, // dwm1001-8 to dwm1001-5
	 2.4, // dwm1001-8 to dwm1001-6
	 2.4, // dwm1001-8 to dwm1001-7
	 6.6822, // dwm1001-9 to dwm1001-1
	 5.9458, // dwm1001-9 to dwm1001-2
	 5.6984, // dwm1001-9 to dwm1001-3
	 6.2363, // dwm1001-9 to dwm1001-4
	 4.9477, // dwm1001-9 to dwm1001-5
	 4.9477, // dwm1001-9 to dwm1001-6
	 2.6833, // dwm1001-9 to dwm1001-7
	 2.6833, // dwm1001-9 to dwm1001-8
	 7.3394, // dwm1001-10 to dwm1001-1
	 6.8883, // dwm1001-10 to dwm1001-2
	 7.0942, // dwm1001-10 to dwm1001-3
	 7.7219, // dwm1001-10 to dwm1001-4
	 6.0, // dwm1001-10 to dwm1001-5
	 6.4622, // dwm1001-10 to dwm1001-6
	 3.6, // dwm1001-10 to dwm1001-7
	 4.3267, // dwm1001-10 to dwm1001-8
	 1.6971, // dwm1001-10 to dwm1001-9
	 8.2624, // dwm1001-11 to dwm1001-1
	 7.4892, // dwm1001-11 to dwm1001-2
	 6.8883, // dwm1001-11 to dwm1001-3
	 7.1405, // dwm1001-11 to dwm1001-4
	 6.4622, // dwm1001-11 to dwm1001-5
	 6.0, // dwm1001-11 to dwm1001-6
	 4.3267, // dwm1001-11 to dwm1001-7
	 3.6, // dwm1001-11 to dwm1001-8
	 1.6971, // dwm1001-11 to dwm1001-9
	 2.4, // dwm1001-11 to dwm1001-10
	 9.2086, // dwm1001-12 to dwm1001-1
	 9.2429, // dwm1001-12 to dwm1001-2
	 9.5494, // dwm1001-12 to dwm1001-3
	 9.8142, // dwm1001-12 to dwm1001-4
	 8.5466, // dwm1001-12 to dwm1001-5
	 9.0379, // dwm1001-12 to dwm1001-6
	 6.246, // dwm1001-12 to dwm1001-7
	 6.9031, // dwm1001-12 to dwm1001-8
	 4.4023, // dwm1001-12 to dwm1001-9
	 3.0926, // dwm1001-12 to dwm1001-10
	 4.2666, // dwm1001-12 to dwm1001-11
	 9.5868, // dwm1001-13 to dwm1001-1
	 9.2122, // dwm1001-13 to dwm1001-2
	 9.2122, // dwm1001-13 to dwm1001-3
	 9.5868, // dwm1001-13 to dwm1001-4
	 8.3167, // dwm1001-13 to dwm1001-5
	 8.4881, // dwm1001-13 to dwm1001-6
	 5.9276, // dwm1001-13 to dwm1001-7
	 6.1657, // dwm1001-13 to dwm1001-8
	 3.5531, // dwm1001-13 to dwm1001-9
	 2.385, // dwm1001-13 to dwm1001-10
	 2.9271, // dwm1001-13 to dwm1001-11
	 2.0809, // dwm1001-13 to dwm1001-12
	 9.8142, // dwm1001-14 to dwm1001-1
	 9.5494, // dwm1001-14 to dwm1001-2
	 9.2429, // dwm1001-14 to dwm1001-3
	 9.2086, // dwm1001-14 to dwm1001-4
	 8.7134, // dwm1001-14 to dwm1001-5
	 8.5466, // dwm1001-14 to dwm1001-6
	 6.4724, // dwm1001-14 to dwm1001-7
	 6.246, // dwm1001-14 to dwm1001-8
	 4.062, // dwm1001-14 to dwm1001-9
	 3.5276, // dwm1001-14 to dwm1001-10
	 3.0926, // dwm1001-14 to dwm1001-11
	 2.4, // dwm1001-14 to dwm1001-12
	 2.0809, // dwm1001-14 to dwm1001-13
};