
#include "nodes.h"
#include "positioning.h"

static uint16_t own_node_id;
static float32_t estimates[NUM_NODES][3];
static float32_t distances[NUM_NODES];

void positioning_init() {
    own_node_id = get_own_node_id();
    positioning_reset();
}

void positioning_reset() {
    // reset all position and distances
    memset(node_pos, 0, sizeof(node_pos));
    memset(distances, 0, sizeof(distances));
}

void positioning_set_distances(float32_t new_distances[NUM_NODES]) {
    memcpy(distances, new_distances, sizeof(distances));
}

void positioning_set_estimate(uint16_t other_number, pos_t &new_pos) {
    node_pos[other_number] = *new_pos;
}

void positioning_update_own_estimate(pos_t &out) {

    estimates

    if own_number == 0 -> skip everything
    if own_number == 1 -> skip two dimensions (e.g. set to the actual distance)
    if own_number == 2 -> skip one dimensions (e.g. set to the actual distance)
    if own_number == 3 -> force one dimension > 0


    // iterate over each dimension
    for(size_t d = 0; d < 3; d++) {

    }



    node_pos[own_node_id]


    // copy estimate back
    *out = node_pos[own_node_id];
}