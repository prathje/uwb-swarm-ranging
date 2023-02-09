#include "nodes.h"
// ----------------------
// Definition of Testbed trento_a
#if NUM_NODES>7
#error Testbed TRENTO_A only has 7 nodes
#elif NUM_NODES<7
#warning Testbed TRENTO_A has more nodes available (7 in total)
#endif
uint16_t node_ids[NUM_NODES] = {
	0x5b9a, // dwm1001-1
	0x56d4, // dwm1001-2
	0x0345, // dwm1001-3
	0x9535, // dwm1001-4
	0x87e8, // dwm1001-5
	0xa7d8, // dwm1001-6
	0x24f1, // dwm1001-7
};
int16_t node_factory_antenna_delay_offsets[NUM_NODES] = {
	22, // dwm1001-1
	9, // dwm1001-2
	22, // dwm1001-3
	22, // dwm1001-4
	8, // dwm1001-5
	10, // dwm1001-6
	10, // dwm1001-7
};
float32_t node_distances[NUM_PAIRS] = {
	 4.1886, // dwm1001-2 to dwm1001-1
	 2.8902, // dwm1001-3 to dwm1001-1
	 3.2404, // dwm1001-3 to dwm1001-2
	 4.1442, // dwm1001-4 to dwm1001-1
	 6.245, // dwm1001-4 to dwm1001-2
	 3.01, // dwm1001-4 to dwm1001-3
	 4.5621, // dwm1001-5 to dwm1001-1
	 8.7045, // dwm1001-5 to dwm1001-2
	 7.0529, // dwm1001-5 to dwm1001-3
	 6.4106, // dwm1001-5 to dwm1001-4
	 3.5417, // dwm1001-6 to dwm1001-1
	 6.9276, // dwm1001-6 to dwm1001-2
	 6.4305, // dwm1001-6 to dwm1001-3
	 7.1249, // dwm1001-6 to dwm1001-4
	 3.0, // dwm1001-6 to dwm1001-5
	 4.7443, // dwm1001-7 to dwm1001-1
	 6.2323, // dwm1001-7 to dwm1001-2
	 7.1752, // dwm1001-7 to dwm1001-3
	 8.8789, // dwm1001-7 to dwm1001-4
	 5.9804, // dwm1001-7 to dwm1001-5
	 2.9806, // dwm1001-7 to dwm1001-6
};