
#include <drivers/uart.h>
#include <zephyr.h>
#include <logging/log.h>
#include "log.h"
#include "nodes.h"

#define LOG_BUF_SIZE (8192)

LOG_MODULE_REGISTER(logging);


static const struct device* uart_device = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

K_SEM_DEFINE(log_buf_sem, 1, 1);

void uart_out(char* msg) {
    k_sem_take(&log_buf_sem, K_FOREVER);
    while (*msg != '\0') {
        uart_poll_out(uart_device, *msg);
        msg++;
    }
    k_sem_give(&log_buf_sem);
}

void uart_disabled(char* msg) {
    //NOOP
}



static char log_buf[LOG_BUF_SIZE] = {0};
static size_t log_buf_count = 0;


void log_out(char* msg) {
    k_sem_take(&log_buf_sem, K_FOREVER);

    while (*msg != '\0' && log_buf_count < LOG_BUF_SIZE) {
        log_buf[log_buf_count] = *msg;
        log_buf_count++;
        msg++;
    }

    k_sem_give(&log_buf_sem);

    if (log_buf_count == LOG_BUF_SIZE-1) {
        LOG_WRN("LOG IS FULL!!!");
    }
}

void log_flush() {
    k_sem_take(&log_buf_sem, K_FOREVER);

    for (size_t i = 0; i < log_buf_count; i++) {
        uart_poll_out(uart_device, log_buf[i]);
    }

    log_buf_count = 0;
    (void)memset(&log_buf, 0, sizeof(log_buf));

    k_sem_give(&log_buf_sem);
}


size_t log_get_count() {
    return log_buf_count;
}