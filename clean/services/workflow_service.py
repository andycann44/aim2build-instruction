from clean.services import analyzer_scan_service, sequence_service, truth_service


def get_set_workflow(set_num: str):
    confirmed_rows = truth_service.get_all_confirmed_bags(set_num)
    confirmed_rows = [row for row in confirmed_rows if "_error" not in row]

    if confirmed_rows:
        sequence = sequence_service.get_sequence_from_truth(set_num)
        return {
            "ok": True,
            "set_num": set_num,
            "mode": "truth_guided",
            "sequence": sequence,
        }

    analyzer_scan = analyzer_scan_service.scan_set_with_analyzer(set_num, include_all=False)
    return {
        "ok": bool(analyzer_scan.get("ok")),
        "set_num": set_num,
        "mode": "fresh_precheck",
        "analyzer_scan": analyzer_scan,
    }
