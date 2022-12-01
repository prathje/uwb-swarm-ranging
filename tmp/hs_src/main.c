#include <logging/log.h>
#include <syscalls/rand32.h>
#include <zephyr.h>

#include "deca_device_api.h"
#include "deca_regs.h"
#include "deca_spi.h"
#include "port.h"
 
#include "host_interface.h"
#include "message_definition.h"
#include "messages.h"
#include "misc.h"
#include "storage.h"
#include "timestamps.h"
#include "typedefs.h"

LOG_MODULE_REGISTER(main);

K_TIMER_DEFINE(transmission_timer, NULL, NULL);

static dwt_config_t config = {5, DWT_PRF_64M, DWT_PLEN_128, DWT_PAC8, 9, 9, 1, DWT_BR_6M8, DWT_PHRMODE_STD, (129)};

static received_message_t received_messages[CONFIG_NUM_PARTICIPANTS];

self_t self = {.id = 0, .sequence_number = 1};

/**
 * @brief Get the id object
 *
 * This is a static part id to ranging id mapping.
 *
 * @return uint16_t the id. If no mapping exists, 0 is returned as default.
 */
ranging_id_t get_id() {
    uint32_t part_id = dwt_getpartid();
    switch (part_id) {
        case 268447190:
            return 1;
        case 268446691:
            return 2;
        case 268447501:
            return 3;
        case 268438754:
            return 4;
        case 268446529:
            return 5;
        case 268447385:
            return 6;
        default:
            return 0;
    }
}

/**
 * @brief Assembles and transmits a message.
 */
void send_message() {
    LOG_DBG("Sending message");
    uint32_t tx_time = (read_systemtime() + (CONFIG_TX_PROCESSING_DELAY * UUS_TO_DWT_TIME)) >> 8;
    timestamp_t tx_timestamp = (((uint64) (tx_time & 0xFFFFFFFEUL)) << 8) + CONFIG_TX_ANTENNA_DELAY;
    dwt_setdelayedtrxtime(tx_time);

    size_t message_buffer_len = TX_TIMESTAMP_IDX + TIMESTAMP_SIZE + CONFIG_NUM_PARTICIPANTS * RX_TIMESTAMP_SIZE + 12;
    uint8_t message_buffer[message_buffer_len];
    size_t message_size = construct_message(message_buffer, message_buffer_len, received_messages, CONFIG_NUM_PARTICIPANTS, self, tx_timestamp);

    // Add two bytes to message length for the DWMs automatic checksum
    dwt_writetxdata(message_size + 2, message_buffer, 0);
    dwt_writetxfctrl(message_size + 2, 0, 1);
    dwt_starttx(DWT_START_TX_DELAYED);

    while (!(dwt_read32bitreg(SYS_STATUS_ID) & SYS_STATUS_TXFRS)) {
        // Busy wait untile transmission is finished
    };
    set_tx_timestamp(self.sequence_number, tx_timestamp);

    // Report sending event to host module
    tx_range_info_t tx_info;
    tx_info.id = self.id;
    tx_info.sequence_number = self.sequence_number;
    tx_info.tx_time = tx_timestamp;
    process_out_message(&tx_info, self.id);

    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS);
    dwt_rxenable(DWT_START_RX_IMMEDIATE);

    LOG_HEXDUMP_DBG(message_buffer, message_size, "Sent data");
}

/**
 * @brief Checks if there are incoming messages and processes them
 *
 */
void check_received_messages() {
    uint32_t status = dwt_read32bitreg(SYS_STATUS_ID);
    if (status & SYS_STATUS_RXFCG) {
        LOG_DBG("Received frame");
        uint32_t frame_length = dwt_read32bitreg(RX_FINFO_ID) & RX_FINFO_RXFL_MASK_1023;
        LOG_DBG("Frame length: %d", frame_length);

        // Write message to buffer
        uint8_t* rx_buffer = (uint8_t*) k_malloc(frame_length);
        dwt_readrxdata(rx_buffer, frame_length, 0);

        // Read reception timestamp
        timestamp_t rx_timestamp = read_rx_timestamp();

        // Send messag info to host
        rx_range_info_t rx_info = analyse_message(rx_buffer, frame_length, rx_timestamp);
        process_in_message(&rx_info, self.id);

        // Store timestamps for future transmissions
        received_messages[rx_info.sender_id].sequence_number = rx_info.sequence_number;
        received_messages[rx_info.sender_id].rx_timestamp = rx_info.rx_time;

        LOG_HEXDUMP_DBG(rx_buffer, frame_length - 2, "Received data");

        k_free(rx_info.timestamps);
        k_free(rx_buffer);

        // Processing is done, reset status register
        dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG);
    } else if (status & SYS_STATUS_ALL_RX_ERR) {
        // Bad frame, reset status register
        LOG_WRN("Reception error");
        dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_ERR);
    }
}

int main(void) {
    // Initialise dwm1001
    reset_DW1000();
    port_set_dw1000_slowrate();
    openspi();
    if (dwt_initialise(DWT_LOADUCODE | DWT_READ_OTP_PID) == DWT_ERROR) {
        LOG_ERR("Failed to initialize dwm");
        return -1;
    }
    port_set_dw1000_fastrate();
    dwt_configure(&config);
    dwt_setrxantennadelay(CONFIG_RX_ANTENNA_DELAY);
    dwt_settxantennadelay(CONFIG_TX_ANTENNA_DELAY);
    dwt_setleds(1);

    self.id = get_id();
    LOG_DBG("Ranging id %d\n", self.id);

    for (int i = 0; i < CONFIG_NUM_PARTICIPANTS; i++) {
        received_messages[i].sender_id = i;
        received_messages[i].sequence_number = 0;
    }

    k_timer_start(&transmission_timer, K_MSEC(CONFIG_RANGING_INTERVAL), K_NO_WAIT);

    while (1) {
        check_received_messages();
        if (k_timer_status_get(&transmission_timer) > 0) {
            send_message();
            self.sequence_number++;
            // TODO: Use random number
            // int deviation = sys_rand32_get() % CONFIG_RANGING_INTERVAL_MAX_DEVIATION;
            int deviation = dwt_getpartid() % CONFIG_RANGING_INTERVAL_MAX_DEVIATION;
            k_timer_start(&transmission_timer, K_MSEC(CONFIG_RANGING_INTERVAL + deviation), K_NO_WAIT);
        }
    }
    return 0;
}