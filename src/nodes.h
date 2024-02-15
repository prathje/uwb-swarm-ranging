#ifndef NODE_NUMBERS_H
#define NODE_NUMBERS_H

#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>

#define NUM_NODES 14



#define PAIRS(X) (((X)*((X)-1))/2)
#define NUM_PAIRS (PAIRS(NUM_NODES))
#define PAIR_INDEX(A,B) (((MIN(A,B)*(MIN(A,B)-1))/2+MAX(A,B)))


extern int16_t node_factory_antenna_delay_offsets[NUM_NODES];
extern uint16_t node_ids[NUM_NODES];
extern float32_t node_distances[NUM_PAIRS];

size_t pair_index(uint16_t a, uint16_t b);
uint16_t get_own_node_id();
int8_t get_node_number(uint16_t node_id);
int8_t get_range_bias_by_rssi(int8_t rssi);


#endif