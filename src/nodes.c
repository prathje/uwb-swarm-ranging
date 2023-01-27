#include <zephyr.h>
#include <drivers/hwinfo.h>
#include "nodes.h"


int16_t node_factory_antenna_delay_offsets[NUM_NODES] = {
       8, 9, 7, 22, 11, 12, 7, 22, 22, 22, -14, -9, 9, -13
};

float32_t node_distances[NUM_PAIRS] = {
        1.8515,
        3.8663, 2.4000,
        4.8000, 3.8663, 1.8515,
        2.7906, 1.1697, 2.0611, 3.6807,
        4.7103, 3.1636, 1.1697, 2.2152, 2.4000,
        4.1928, 3.3407, 3.7470, 4.8311, 2.4000, 3.3941,
        5.6550, 4.4497, 3.3407, 3.8340, 3.3941, 2.4000, 2.4000,
        6.6822, 5.9458, 5.6984, 6.2363, 4.9477, 4.9477, 2.6833, 2.6833,
        7.3394, 6.8883, 7.0942, 7.7219, 6.0000, 6.4622, 3.6000, 4.3267, 1.6971,
        8.2624, 7.4892, 6.8883, 7.1405, 6.4622, 6.0000, 4.3267, 3.6000, 1.6971, 2.4000,
        9.2086, 9.2429, 9.5494, 9.8142, 8.5466, 9.0379, 6.2460, 6.9031, 4.4023, 3.0926, 4.2666,
        9.5868, 9.2122, 9.2122, 9.5868, 8.3167, 8.4881, 5.9276, 6.1657, 3.5531, 2.3850, 2.9271, 2.0809,
        9.8142, 9.5494, 9.2429, 9.2086, 8.7134, 8.5466, 6.4724, 6.2460, 4.0620, 3.5276, 3.0926, 2.4000, 2.0809
};

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

    uint16_t node_ids[14] = {
        0x6b93, // c1 05 0a d2 ca 32
        0xf1c3, // 2c 8e 38 0e 2c 2f
        0xc240, // a6 e9 82 fe b9 f2
        0x013f, // b2 04 f6 33 7c 79
        0xb227, // e1 bc c5 19 92 66
        0x033b, // 5e 2a 24 f7 5d 92
        0xf524, // d8 8d 45 48 12 b9
        0x37b2, // 12 6a 1d 03 69 47
        0x15ef, // 2e fb cb ec 8d af
        0x7e02, // c2 7f f2 eb 6d e0
        0xad36, // fa 0e a0 b9 19 b7
        0x3598, // 3c 33 73 36 cb 47
        0x47e0, // c9 65 a0 9b 5b 9f
        0x0e92  // 06 79 a6 59 fe 98
    };

    for (int i = 0; i < NUM_NODES && i < sizeof(node_ids)/sizeof(node_ids[0]); i++) {
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