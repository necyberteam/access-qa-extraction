"""Tests for factoid Q&A generation from templates.

Tests the generate_factoid_pairs() function for each domain,
covering normal generation, boolean templates (yes + no), missing
fields graceful skip, citation presence, and granularity metadata.
"""

from access_qa_extraction.generators.factoids import generate_factoid_pairs

# --- Sample entity data matching what each extractor's clean step produces ---

COMPUTE_RESOURCE_DATA = {
    "name": "Delta",
    "description": "Delta is a GPU-focused HPC system at NCSA. It supports AI workloads.",
    "organization_names": ["NCSA"],
    "feature_names": ["GPU Acceleration", "Container Support"],
    "hasGpu": True,
    "resourceType": "Compute",
    "accessAllocated": True,
    "hardware": {
        "gpus": [{"name": "NVIDIA A100 40GB", "type": "GPU", "details": "..."}],
        "compute_nodes": [],
    },
}

SOFTWARE_DATA = {
    "name": "Python",
    "description": "General-purpose programming language",
    "software_type": "Interpreted Language",
    "available_on_resources": [
        {"resource_id": "delta.ncsa.access-ci.org", "name": "Delta"},
        {"resource_id": "bridges2.psc.access-ci.org", "name": "Bridges-2"},
    ],
    "versions": [
        {"version": "3.11.5", "resource": "delta"},
        {"version": "3.10.12", "resource": "bridges2"},
    ],
    "example_use": "module load python/3.11",
}

ALLOCATION_DATA = {
    "title": "ML for Climate Prediction",
    "pi": "John Doe",
    "institution": "MIT",
    "field_of_science": "Computer Science",
    "allocation_type": "Research",
    "abstract": "This project uses ML to improve climate models.",
    "beginDate": "2024-01-01",
    "endDate": "2025-12-31",
    "resources": [
        {"name": "Delta GPU", "units": "GPU Hours", "allocation": 50000},
        {"name": "Expanse", "units": "SUs", "allocation": 100000},
    ],
}

NSF_AWARD_DATA = {
    "title": "CI CoE: Democratizing ACCESS to CI",
    "principal_investigator": "Dr. John Doe",
    "institution": "U of Illinois",
    "total_intended_award": "$4,500,000",
    "primary_program": "OAC",
    "startDate": "2024-10-01",
    "endDate": "2029-09-30",
    "co_pis": ["Dr. Jane Smith", "Dr. Bob Jones"],
    "award_number": "2450552",
}

AFFINITY_GROUP_DATA = {
    "name": "GPU Computing",
    "description": "A community for GPU computing enthusiasts.",
    "coordinator": "Jane Smith",
    "category": "Technology",
    "slack_link": "https://slack.example.com/gpu",
    "support_url": "https://support.example.com/gpu",
    "upcoming_events": [{"title": "GPU Workshop 2025", "date": "2025-03-15"}],
    "knowledge_base_topics": ["Getting Started with GPUs"],
}


class TestComputeResourcesFactoids:
    """Test factoid generation for compute-resources domain."""

    def test_generates_factoid_pairs(self):
        pairs = generate_factoid_pairs(
            "compute-resources", "delta.ncsa.access-ci.org", COMPUTE_RESOURCE_DATA
        )
        assert len(pairs) == 7  # All 7 templates should fire
        assert all(p.metadata.granularity == "factoid" for p in pairs)
        assert all(p.domain == "compute-resources" for p in pairs)

    def test_ids_use_fq_prefix(self):
        pairs = generate_factoid_pairs(
            "compute-resources", "delta.ncsa.access-ci.org", COMPUTE_RESOURCE_DATA
        )
        ids = [p.id for p in pairs]
        assert "compute-resources_delta.ncsa.access-ci.org_fq_resource_type" in ids
        assert "compute-resources_delta.ncsa.access-ci.org_fq_has_gpu" in ids
        assert "compute-resources_delta.ncsa.access-ci.org_fq_description" in ids

    def test_all_pairs_have_citations(self):
        pairs = generate_factoid_pairs(
            "compute-resources", "delta.ncsa.access-ci.org", COMPUTE_RESOURCE_DATA
        )
        for pair in pairs:
            assert pair.metadata.has_citation is True
            answer = pair.messages[1].content
            assert "<<SRC:compute-resources:delta.ncsa.access-ci.org>>" in answer

    def test_boolean_template_true(self):
        pairs = generate_factoid_pairs(
            "compute-resources", "delta.ncsa.access-ci.org", COMPUTE_RESOURCE_DATA
        )
        gpu_pair = next(p for p in pairs if "fq_has_gpu" in p.id)
        assert "Yes" in gpu_pair.messages[1].content

    def test_boolean_template_false(self):
        no_gpu = {**COMPUTE_RESOURCE_DATA, "hasGpu": False}
        pairs = generate_factoid_pairs("compute-resources", "some.resource", no_gpu)
        gpu_pair = next(p for p in pairs if "fq_has_gpu" in p.id)
        assert "No" in gpu_pair.messages[1].content

    def test_missing_fields_skipped(self):
        minimal = {"name": "Delta"}
        pairs = generate_factoid_pairs("compute-resources", "delta.ncsa.access-ci.org", minimal)
        # Only fq_has_gpu and fq_allocated fire (both just need "name" required)
        # plus fq_has_gpu bool check uses hasGpu (not in data → falsy → "No")
        # fq_allocated bool check uses accessAllocated (not in data → falsy → "No")
        ids = [p.id for p in pairs]
        assert "compute-resources_delta.ncsa.access-ci.org_fq_has_gpu" in ids
        assert "compute-resources_delta.ncsa.access-ci.org_fq_allocated" in ids
        # These should NOT be present (missing required fields)
        assert "compute-resources_delta.ncsa.access-ci.org_fq_resource_type" not in ids
        assert "compute-resources_delta.ncsa.access-ci.org_fq_operator" not in ids

    def test_source_data_is_none(self):
        """Factoid pairs should have source_data=None to avoid JSONL bloat."""
        pairs = generate_factoid_pairs(
            "compute-resources", "delta.ncsa.access-ci.org", COMPUTE_RESOURCE_DATA
        )
        for pair in pairs:
            assert pair.metadata.source_data is None


class TestSoftwareDiscoveryFactoids:
    """Test factoid generation for software-discovery domain."""

    def test_generates_factoid_pairs(self):
        pairs = generate_factoid_pairs("software-discovery", "python", SOFTWARE_DATA)
        assert len(pairs) == 7  # All 7 templates should fire
        assert all(p.metadata.granularity == "factoid" for p in pairs)

    def test_derived_fields(self):
        pairs = generate_factoid_pairs("software-discovery", "python", SOFTWARE_DATA)
        # Check resource count
        count_pair = next(p for p in pairs if "fq_resource_count" in p.id)
        assert "2 ACCESS resources" in count_pair.messages[1].content

        # Check version count
        version_pair = next(p for p in pairs if "fq_version_count" in p.id)
        assert "2 versions" in version_pair.messages[1].content

    def test_boolean_example_use(self):
        pairs = generate_factoid_pairs("software-discovery", "python", SOFTWARE_DATA)
        example_pair = next(p for p in pairs if "fq_has_example" in p.id)
        assert "Yes" in example_pair.messages[1].content

    def test_no_example_use(self):
        no_example = {k: v for k, v in SOFTWARE_DATA.items() if k != "example_use"}
        pairs = generate_factoid_pairs("software-discovery", "python", no_example)
        example_pair = next(p for p in pairs if "fq_has_example" in p.id)
        assert "No" in example_pair.messages[1].content


class TestAllocationsFactoids:
    """Test factoid generation for allocations domain."""

    def test_generates_factoid_pairs(self):
        pairs = generate_factoid_pairs("allocations", "TG-CIS210014", ALLOCATION_DATA)
        assert len(pairs) == 8  # All 8 templates should fire

    def test_resource_list(self):
        pairs = generate_factoid_pairs("allocations", "TG-CIS210014", ALLOCATION_DATA)
        res_pair = next(p for p in pairs if "fq_resource_list" in p.id)
        assert "Delta GPU" in res_pair.messages[1].content
        assert "Expanse" in res_pair.messages[1].content


class TestNSFAwardsFactoids:
    """Test factoid generation for nsf-awards domain."""

    def test_generates_factoid_pairs(self):
        pairs = generate_factoid_pairs("nsf-awards", "2450552", NSF_AWARD_DATA)
        assert len(pairs) == 8  # All 8 templates should fire

    def test_copis_yes(self):
        pairs = generate_factoid_pairs("nsf-awards", "2450552", NSF_AWARD_DATA)
        copi_pair = next(p for p in pairs if "fq_has_copis" in p.id)
        assert "Yes" in copi_pair.messages[1].content
        assert "2 co-PI(s)" in copi_pair.messages[1].content

    def test_copis_no(self):
        no_copis = {**NSF_AWARD_DATA, "co_pis": []}
        pairs = generate_factoid_pairs("nsf-awards", "9876543", no_copis)
        copi_pair = next(p for p in pairs if "fq_has_copis" in p.id)
        assert "No" in copi_pair.messages[1].content


class TestAffinityGroupsFactoids:
    """Test factoid generation for affinity-groups domain."""

    def test_generates_factoid_pairs(self):
        pairs = generate_factoid_pairs("affinity-groups", "42", AFFINITY_GROUP_DATA)
        assert len(pairs) == 6  # All 6 templates should fire

    def test_boolean_slack_yes(self):
        pairs = generate_factoid_pairs("affinity-groups", "42", AFFINITY_GROUP_DATA)
        slack_pair = next(p for p in pairs if "fq_has_slack" in p.id)
        assert "Yes" in slack_pair.messages[1].content

    def test_boolean_slack_no(self):
        no_slack = {k: v for k, v in AFFINITY_GROUP_DATA.items() if k != "slack_link"}
        pairs = generate_factoid_pairs("affinity-groups", "42", no_slack)
        slack_pair = next(p for p in pairs if "fq_has_slack" in p.id)
        assert "No" in slack_pair.messages[1].content


class TestUnknownDomain:
    """Test graceful handling of unknown domains."""

    def test_unknown_domain_returns_empty(self):
        pairs = generate_factoid_pairs("unknown-domain", "123", {"name": "test"})
        assert pairs == []


# ── Data Quality Guards ─────────────────────────────────────────────


class TestComputeResourcesQualityGuards:
    """Test that empty/junk data doesn't produce broken factoid answers."""

    def test_empty_org_list_skips_operator(self):
        data = {**COMPUTE_RESOURCE_DATA, "organization_names": []}
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        ids = [p.id for p in pairs]
        assert "compute-resources_delta_fq_operator" not in ids

    def test_org_list_with_empty_string_skips_operator(self):
        data = {**COMPUTE_RESOURCE_DATA, "organization_names": [""]}
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        ids = [p.id for p in pairs]
        assert "compute-resources_delta_fq_operator" not in ids

    def test_partial_org_list_filters_empty(self):
        data = {**COMPUTE_RESOURCE_DATA, "organization_names": ["NCSA", "", " "]}
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        op_pair = next(p for p in pairs if "fq_operator" in p.id)
        assert "NCSA" in op_pair.messages[1].content
        assert ", ," not in op_pair.messages[1].content
        assert ", ." not in op_pair.messages[1].content

    def test_junk_feature_names_filtered(self):
        data = {**COMPUTE_RESOURCE_DATA, "feature_names": ["Unknown Type"]}
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        ids = [p.id for p in pairs]
        assert "compute-resources_delta_fq_features" not in ids

    def test_mixed_features_filters_unknown(self):
        data = {
            **COMPUTE_RESOURCE_DATA,
            "feature_names": ["GPU Acceleration", "Unknown Type", "Container Support"],
        }
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        feat_pair = next(p for p in pairs if "fq_features" in p.id)
        assert "GPU Acceleration" in feat_pair.messages[1].content
        assert "Container Support" in feat_pair.messages[1].content
        assert "Unknown" not in feat_pair.messages[1].content

    def test_empty_gpu_list_skips_gpu_model(self):
        data = {**COMPUTE_RESOURCE_DATA, "hardware": {"gpus": [], "compute_nodes": []}}
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        ids = [p.id for p in pairs]
        assert "compute-resources_delta_fq_gpu_model" not in ids

    def test_gpu_with_empty_name_skips(self):
        data = {
            **COMPUTE_RESOURCE_DATA,
            "hardware": {"gpus": [{"name": "", "type": "GPU"}], "compute_nodes": []},
        }
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        ids = [p.id for p in pairs]
        assert "compute-resources_delta_fq_gpu_model" not in ids

    def test_whitespace_description_skips(self):
        data = {**COMPUTE_RESOURCE_DATA, "description": "   "}
        pairs = generate_factoid_pairs("compute-resources", "delta", data)
        ids = [p.id for p in pairs]
        assert "compute-resources_delta_fq_description" not in ids


class TestSoftwareQualityGuards:
    """Test quality guards for software-discovery factoids."""

    def test_empty_version_skips_latest(self):
        data = {**SOFTWARE_DATA, "versions": [{"version": ""}]}
        pairs = generate_factoid_pairs("software-discovery", "python", data)
        ids = [p.id for p in pairs]
        assert "software-discovery_python_fq_latest_version" not in ids

    def test_empty_resource_dicts_skip(self):
        data = {**SOFTWARE_DATA, "available_on_resources": [{}]}
        pairs = generate_factoid_pairs("software-discovery", "python", data)
        ids = [p.id for p in pairs]
        assert "software-discovery_python_fq_resource_list" not in ids
        # Count should be 0 → template fires with "0 ACCESS resources" (valid)
        count_pair = next((p for p in pairs if "fq_resource_count" in p.id), None)
        if count_pair:
            assert "0 ACCESS resources" in count_pair.messages[1].content


class TestAllocationsQualityGuards:
    """Test quality guards for allocations factoids."""

    def test_empty_resource_names_skip(self):
        data = {**ALLOCATION_DATA, "resources": [{"name": ""}, {"name": ""}]}
        pairs = generate_factoid_pairs("allocations", "TG-123", data)
        ids = [p.id for p in pairs]
        assert "allocations_TG-123_fq_resource_list" not in ids
        # resource_count should be 0 after filtering
        count_pair = next((p for p in pairs if "fq_resource_count" in p.id), None)
        if count_pair:
            assert "0 resources" in count_pair.messages[1].content


class TestNSFAwardsQualityGuards:
    """Test quality guards for nsf-awards factoids."""

    def test_empty_copis_count_zero(self):
        data = {**NSF_AWARD_DATA, "co_pis": ["", " "]}
        pairs = generate_factoid_pairs("nsf-awards", "1234567", data)
        copi_pair = next(p for p in pairs if "fq_has_copis" in p.id)
        # Filtered co_pis is empty → bool falsy → "No" answer
        assert "No" in copi_pair.messages[1].content

    def test_partial_copis_filters(self):
        data = {**NSF_AWARD_DATA, "co_pis": ["Dr. Jane Smith", "", "Dr. Bob Jones"]}
        pairs = generate_factoid_pairs("nsf-awards", "1234567", data)
        copi_pair = next(p for p in pairs if "fq_has_copis" in p.id)
        assert "2 co-PI(s)" in copi_pair.messages[1].content
        assert "Dr. Jane Smith" in copi_pair.messages[1].content


class TestQualityDefectDetection:
    """Test the _has_quality_defect post-format validation."""

    def test_zero_value_still_fires(self):
        """resource_count: 0 is valid data, should produce a factoid."""
        data = {**SOFTWARE_DATA, "available_on_resources": []}
        pairs = generate_factoid_pairs("software-discovery", "python", data)
        count_pair = next((p for p in pairs if "fq_resource_count" in p.id), None)
        assert count_pair is not None
        assert "0 ACCESS resources" in count_pair.messages[1].content

    def test_valid_pair_passes_quality_check(self):
        """Normal data should pass quality checks without issue."""
        pairs = generate_factoid_pairs(
            "compute-resources", "delta", COMPUTE_RESOURCE_DATA
        )
        assert len(pairs) == 7  # Same as before quality guards were added
