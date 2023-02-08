from base import get_dist

import testbed.lille as lille
import testbed.trento_a as trento_a
import testbed.trento_b as trento_b

import sys

DEFAULT_DELAY = 16450

TESTBEDS = [
    lille, trento_a, trento_b
]

def print_testbed(t):
    n = len(t.dev_positions.keys())

    print("// ----------------------")
    print("// Definition of Testbed {}".format(t.name))

    print("#if NUM_NODES>{}".format(n))
    print("#error Testbed {} only has {} nodes".format(t.name.upper(), n))
    print("#elif NUM_NODES<{}".format(n))
    print("#warning Testbed {} has more nodes available ({} in total)".format(t.name.upper(), n))
    print("#endif")

    print("uint16_t node_ids[NUM_NODES] = {")
    for d in t.devs:
        print("\t{}, // {}".format(t.dev_ids[d],  d))
    print("};")

    print("int16_t node_factory_antenna_delay_offsets[NUM_NODES] = {")
    for d in t.devs:
        print("\t{}, // {}".format(t.factory_delays[d] - DEFAULT_DELAY, d))
    print("};")

    print("float32_t node_distances[NUM_PAIRS] = {")

    for (a, da) in enumerate(t.devs):
        for (b, db) in enumerate(t.devs):
            if b < a:
                print("\t {}, // {} to {}".format(round(get_dist(t.dev_positions[da], t.dev_positions[db]), 4), da, db))
    print("};")




def print_all_testbeds():
    for (i, t) in enumerate(TESTBEDS):
        if i == 0:
            print("#if TESTBED==TESTBED_{}".format(t.name.upper()))
        else:
            print("#elif TESTBED==TESTBED_{}".format(t.name.upper()))
        print_testbed(t)
    print("#endif")



if __name__ == "__main__":

    if len(sys.argv) > 1:
        found = False
        for t in TESTBEDS:
            if t.name == sys.argv[1]:
                print("#include \"nodes.h\"")
                print_testbed(t)
                found = True
        if not found:
            print("Could not find testbed {}".format(sys.argv[1]))
    else:
        print("#include \"nodes.h\"")
        print_all_testbeds()