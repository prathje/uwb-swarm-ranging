#include <zephyr.h>
#include <drivers/hwinfo.h>
#include "nodes.h"


// testbed defined in testbed.c

inline size_t pair_index(uint16_t a, uint16_t b) {
       if (a > b) {
            size_t num_pairs_before = (a*(a-1))/2; // we have exactly that sum of measurements before
            return num_pairs_before+b;
       } else {
            return pair_index(b, a);
       }
}


uint16_t get_own_node_id() {
    uint8_t id_buf[2] = {0};
    ssize_t copied_bytes = hwinfo_get_device_id(id_buf, sizeof(id_buf));
    return id_buf[0] << 8 | id_buf[1];
}


int8_t get_node_number(uint16_t node_id) {
    for (int i = 0; i < NUM_NODES; i++) {
        if (node_id == node_ids[i]) {
            return i;
        }
    }

    return -1;  //node number not found!
}

#define RANGE_CORR_MAX_RSSI (-61)
#define RANGE_CORR_MIN_RSSI (-93)

int8_t range_bias_by_rssi[RANGE_CORR_MAX_RSSI-RANGE_CORR_MIN_RSSI+1] = {
    -23, // -61dBm (-11 cm)
    -23, // -62dBm (-10.75 cm)
    -22, // -63dBm (-10.5 cm)
    -22, // -64dBm (-10.25 cm)
    -21, // -65dBm (-10.0 cm)
    -21, // -66dBm (-9.65 cm)
    -20, // -67dBm (-9.3 cm)
    -19, // -68dBm (-8.75 cm)
    -17, // -69dBm (-8.2 cm)
    -16, // -70dBm (-7.55 cm)
    -15, // -71dBm (-6.9 cm)
    -13, // -72dBm (-6.0 cm)
    -11, // -73dBm (-5.1 cm)
    -8, // -74dBm (-3.9 cm)
    -6, // -75dBm (-2.7 cm)
    -3, // -76dBm (-1.35 cm)
    0, // -77dBm (0.0 cm)
    2, // -78dBm (1.05 cm)
    4, // -79dBm (2.1 cm)
    6, // -80dBm (2.8 cm)
    7, // -81dBm (3.5 cm)
    8, // -82dBm (3.85 cm)
    9, // -83dBm (4.2 cm)
    10, // -84dBm (4.55 cm)
    10, // -85dBm (4.9 cm)
    12, // -86dBm (5.55 cm)
    13, // -87dBm (6.2 cm)
    14, // -88dBm (6.65 cm)
    15, // -89dBm (7.1 cm)
    16, // -90dBm (7.35 cm)
    16, // -91dBm (7.6 cm)
    17, // -92dBm (7.85 cm)
    17, // -93dBm (8.1 cm)
};

int8_t get_range_bias_by_rssi(int8_t rssi) {
    rssi = MAX(MIN(RANGE_CORR_MAX_RSSI, rssi), RANGE_CORR_MIN_RSSI);
    return range_bias_by_rssi[-(rssi-RANGE_CORR_MAX_RSSI)];
}