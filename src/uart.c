
#include <drivers/uart.h>
#include "uart.h"

static const struct device* uart_device = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

void uart_out(char* msg) {
    while (*msg != '\0') {
        uart_poll_out(uart_device, *msg);
        msg++;
    }
}

void uart_disabled(char* msg) {
    //NOOP
}