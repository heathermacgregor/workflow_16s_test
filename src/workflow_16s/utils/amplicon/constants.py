# workflow_16s/utils/amplicon/constants.py


COMPREHENSIVE_V_REGIONS = {
    "V1": {"fwd_pos": 69, "rev_pos": 99, "leeway": 40},
    "V2": {"fwd_pos": 137, "rev_pos": 242, "leeway": 40},
    "V3": {"fwd_pos": 433, "rev_pos": 497, "leeway": 40},
    "V4": {"fwd_pos": 576, "rev_pos": 682, "leeway": 50},
    "V5": {"fwd_pos": 822, "rev_pos": 879, "leeway": 40},
    "V6": {"fwd_pos": 986, "rev_pos": 1043, "leeway": 40},
    "V7": {"fwd_pos": 1117, "rev_pos": 1173, "leeway": 40},
    "V8": {"fwd_pos": 1243, "rev_pos": 1294, "leeway": 40},
    "V9": {"fwd_pos": 1435, "rev_pos": 1465, "leeway": 40},
    "V1-V2": {"fwd_pos": 27, "rev_pos": 338, "leeway": 40},
    "V1-V3": {"fwd_pos": 27, "rev_pos": 534, "leeway": 50},
    "V2-V3": {"fwd_pos": 338, "rev_pos": 534, "leeway": 50},
    "V3-V4": {"fwd_pos": 341, "rev_pos": 805, "leeway": 50},
    "V4-V5": {"fwd_pos": 515, "rev_pos": 926, "leeway": 50},
    "V5-V7": {"fwd_pos": 785, "rev_pos": 1100, "leeway": 60},
    "V6-V8": {"fwd_pos": 926, "rev_pos": 1392, "leeway": 75},
    "V7-V9": {"fwd_pos": 1100, "rev_pos": 1492, "leeway": 100},
    "Full-Length": {"fwd_pos": 27, "rev_pos": 1492, "leeway": 100},
}