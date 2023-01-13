#include <zephyr.h>
#include <drivers/hwinfo.h>



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

    for (int i = 0; i < sizeof(node_ids)/sizeof(node_ids[0]); i++) {
        if (node_id == node_ids[i]) {
            return i;
        }
    }

    return -1;  //node number not found!
}