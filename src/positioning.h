#if 0
#ifndef POSITIONING_H
#define POSITIONING_H

typedef float32_t pos_t[3]:

void positioning_init();
void positioning_reset();

void positioning_set_distances(float32_t distances[NUM_NODES]);
void positioning_set_estimate(uint16_t other_number, pos_t &new_pos);
void positioning_update_own_estimate(pos_t &out);

#endif
#endif