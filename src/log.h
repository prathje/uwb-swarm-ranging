#ifndef LOG_H
#define LOG_H


void uart_out(char* msg);
void uart_disabled(char* msg);

void log_out(char* msg);
void log_flush();

#endif