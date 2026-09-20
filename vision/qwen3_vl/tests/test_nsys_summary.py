# SPDX-License-Identifier: Apache-2.0
import sqlite3

import pytest
from vision.qwen3_vl.nsys_decoder_summary import timeline_summary


def test_timeline_counts_overlapping_work_once_and_clips_copies():
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000000,4000000),(2000000,3000000),(6000000,8000000);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (4000000,5000000,1,24),(7500000,9000000,2,8);
    """)
    summary = timeline_summary(connection)
    assert summary["span_ms"] == 7
    assert summary["kernel_idle_percent"] == pytest.approx(200 / 7)
    assert summary["kernel_and_copy_idle_ms"] == 1
    assert summary["kernel_and_copy_idle_percent"] == pytest.approx(100 / 7)
    assert summary["runtime_apis"] == []
    clipped = timeline_summary(connection, (2000000, 7000000))
    assert clipped["span_ms"] == 5
    assert clipped["kernel_and_copy_idle_percent"] == 20
