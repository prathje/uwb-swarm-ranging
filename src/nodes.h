#ifndef NODE_NUMBERS_H
#define NODE_NUMBERS_H

#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>

#define NUM_NODES 7



#define PAIRS(X) (((X)*((X)-1))/2)
#define NUM_PAIRS (PAIRS(NUM_NODES))

extern int16_t node_factory_antenna_delay_offsets[NUM_NODES];
extern uint16_t node_ids[NUM_NODES];
extern float32_t node_distances[NUM_PAIRS];
#if 1
extern float32_t estimation_mat_full[NUM_NODES][(NUM_PAIRS/2)];
#else
extern float32_t estimation_mat_robust[NUM_NODES-1][(NUM_NODES-1)*(NUM_NODES-2)/2)];
#endif


size_t pair_index(uint16_t a, uint16_t b);
uint16_t get_own_node_id();
int8_t get_node_number(uint16_t node_id);

int8_t get_range_bias_by_rssi(int8_t rssi);


#endif