#ifndef MSG_H
#define MSG_H


/*static int format_msg_to_json(char *buf, size_t buf_len, struct msg_ts msg_ts_arr[], size_t num_ts) {
    char *c = buf;
    size_t remaining_len = buf_len;

    if (num_ts == 0) {
        return snprintf(c, remaining_len, "{}");
    }

    int ret = 0;
    // write open parentheses
    {
        ret = snprintf(c, remaining_len, "{ \"tx\": ");
        if (ret < 0) { return ret; }
        c += ret;
        remaining_len -= ret;
    }

    // write tx ts
    {
        ret = format_msg_ts_to_json(c, remaining_len, &msg_ts_arr[0]);
        if (ret < 0) { return ret; }
        c += ret;
        remaining_len -= ret;
    }

    // prepare rx ts parentheses
    {
        ret = snprintf(c, remaining_len, ", \"rx\": [");
        if (ret < 0) { return ret; }
        c += ret;
        remaining_len -= ret;
    }

    // write message content
    for(int i = 1; i < num_ts; i++) {
        // add separators in between
        if (i > 1)
        {
            ret = snprintf(c, remaining_len, ", ");
            if (ret < 0) { return ret; }
            c += ret;
            remaining_len -= ret;
        }

        // write ts content
        {
            ret = format_msg_ts_to_json(c, remaining_len, &msg_ts_arr[i]);
            if (ret < 0) { return ret; }
            c += ret;
            remaining_len -= ret;
        }
    }

    // write close parentheses
    {
        ret = snprintf(c, remaining_len, "]}");
        if (ret < 0) { return ret; }
        //c += ret;
        //remaining_len -= ret;
    }

    return 0;
}*/
#endif