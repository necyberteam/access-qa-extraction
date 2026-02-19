"""Tests for ComparisonGenerator across all 5 domains.

Tests the cross-entity comparison Q&A generation, including:
- Existing compute-resources comparisons (GPU, feature, organization)
- Existing software-discovery comparisons (availability, cross-domain)
- New allocations comparisons (by FOS, institution, resource)
- New nsf-awards comparisons (by program, institution)
- New affinity-groups comparisons (by category)
- Count+sample formatting for large domains
- Minimum group size thresholds
- Granularity metadata on all comparison pairs
"""

from access_qa_extraction.generators.comparisons import ComparisonGenerator

# --- Sample data matching what each extractor populates in raw_data ---

COMPUTE_DATA = {
    "delta.ncsa.access-ci.org": {
        "name": "Delta",
        "has_gpu": True,
        "gpu_types": ["NVIDIA A100"],
        "features": ["GPU Acceleration", "Container Support"],
        "organizations": ["NCSA"],
    },
    "bridges2.psc.access-ci.org": {
        "name": "Bridges-2",
        "has_gpu": True,
        "gpu_types": ["NVIDIA V100"],
        "features": ["GPU Acceleration", "Container Support"],
        "organizations": ["PSC"],
    },
    "expanse.sdsc.access-ci.org": {
        "name": "Expanse",
        "has_gpu": True,
        "gpu_types": ["NVIDIA V100"],
        "features": ["GPU Acceleration"],
        "organizations": ["SDSC"],
    },
}

SOFTWARE_DATA = {
    "pytorch": {
        "name": "PyTorch",
        "resources": ["delta.ncsa.access-ci.org", "bridges2.psc.access-ci.org"],
        "software_type": "ML Framework",
    },
    "tensorflow": {
        "name": "TensorFlow",
        "resources": [
            "delta.ncsa.access-ci.org",
            "bridges2.psc.access-ci.org",
            "expanse.sdsc.access-ci.org",
        ],
        "software_type": "ML Framework",
    },
    "gcc": {
        "name": "GCC",
        "resources": ["delta.ncsa.access-ci.org"],
        "software_type": "Compiler",
    },
}

ALLOCATIONS_DATA = {
    "TG-CIS210014": {
        "name": "ML for Climate",
        "project_id": "TG-CIS210014",
        "pi": "John Doe",
        "institution": "MIT",
        "fos": "Computer Science",
        "allocation_type": "Research",
        "resource_count": 2,
        "resource_names": ["Delta GPU", "Expanse"],
    },
    "TG-CIS220001": {
        "name": "Deep Learning Methods",
        "project_id": "TG-CIS220001",
        "pi": "Jane Smith",
        "institution": "Stanford",
        "fos": "Computer Science",
        "allocation_type": "Research",
        "resource_count": 1,
        "resource_names": ["Delta GPU"],
    },
    "TG-BIO230001": {
        "name": "Protein Folding",
        "project_id": "TG-BIO230001",
        "pi": "Bob Jones",
        "institution": "MIT",
        "fos": "Biophysics",
        "allocation_type": "Research",
        "resource_count": 1,
        "resource_names": ["Bridges-2"],
    },
    "TG-CIS230002": {
        "name": "NLP Research",
        "project_id": "TG-CIS230002",
        "pi": "Alice Chen",
        "institution": "MIT",
        "fos": "Computer Science",
        "allocation_type": "Research",
        "resource_count": 1,
        "resource_names": ["Delta GPU"],
    },
}

NSF_AWARDS_DATA = {
    "2345678": {
        "name": "Advanced Computing for Climate",
        "award_number": "2345678",
        "pi": "Dr. Jane Smith",
        "institution": "MIT",
        "primary_program": "OAC",
        "has_co_pis": True,
    },
    "9876543": {
        "name": "HPC for Genomics",
        "award_number": "9876543",
        "pi": "Dr. Mike Lee",
        "institution": "Stanford",
        "primary_program": "OAC",
        "has_co_pis": False,
    },
    "1111111": {
        "name": "AI for Materials Science",
        "award_number": "1111111",
        "pi": "Dr. Alice Chen",
        "institution": "MIT",
        "primary_program": "OAC",
        "has_co_pis": True,
    },
    "2222222": {
        "name": "Quantum Computing Research",
        "award_number": "2222222",
        "pi": "Dr. Bob Jones",
        "institution": "Caltech",
        "primary_program": "PHY",
        "has_co_pis": False,
    },
}

AFFINITY_GROUPS_DATA = {
    "1": {
        "name": "GPU Computing",
        "category": "Technology",
    },
    "2": {
        "name": "Machine Learning",
        "category": "Technology",
    },
    "3": {
        "name": "Women in HPC",
        "category": "Community",
    },
    "4": {
        "name": "Cloud Computing",
        "category": "Technology",
    },
}


class TestComputeResourcesComparisons:
    """Test compute-resources comparison generation (existing)."""

    def test_gpu_availability(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, {})

        gpu_pairs = [p for p in pairs if p.id.startswith("cmp_gpu_")]
        # V100 appears on 2 resources (Bridges-2, Expanse) → should generate
        assert any("V100" in p.messages[0].content for p in gpu_pairs)

    def test_feature_availability(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, {})

        feat_pairs = [p for p in pairs if p.id.startswith("cmp_feat_")]
        # GPU Acceleration appears on 3 resources → should generate
        assert any("gpu acceleration" in p.messages[0].content.lower() for p in feat_pairs)
        # Container Support appears on 2 resources → should generate
        assert any("container support" in p.messages[0].content.lower() for p in feat_pairs)

    def test_all_pairs_have_comparison_granularity(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, SOFTWARE_DATA)
        for pair in pairs:
            assert pair.metadata.granularity == "comparison"

    def test_all_pairs_have_citations(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, SOFTWARE_DATA)
        for pair in pairs:
            assert pair.metadata.has_citation is True
            assert "<<SRC:" in pair.messages[1].content


class TestSoftwareComparisons:
    """Test software-discovery comparison generation (existing)."""

    def test_software_availability(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, SOFTWARE_DATA)

        sw_pairs = [p for p in pairs if p.id.startswith("cmp_sw_")]
        # PyTorch on 2 resources → should generate
        assert any("PyTorch" in p.messages[0].content for p in sw_pairs)
        # TensorFlow on 3 resources → should generate
        assert any("TensorFlow" in p.messages[0].content for p in sw_pairs)
        # GCC on 1 resource → should NOT generate (below min)
        assert not any("GCC" in p.messages[0].content for p in sw_pairs)

    def test_cross_domain_gpu_software(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, SOFTWARE_DATA)

        cross_pairs = [p for p in pairs if p.id.startswith("cmp_gpu_sw_")]
        # PyTorch on GPU resources (Delta has A100, Bridges-2 has V100 — both GPU)
        assert any("PyTorch" in p.messages[0].content for p in cross_pairs)


class TestAllocationsComparisons:
    """Test allocations comparison generation (new)."""

    def test_allocations_by_fos(self):
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, allocations_data=ALLOCATIONS_DATA)

        fos_pairs = [p for p in pairs if p.id.startswith("cmp_alloc_fos")]
        # Computer Science has 3 projects → should generate (≥ MIN_GROUP_SIZE_LARGE=3)
        assert len(fos_pairs) == 1
        assert "Computer Science" in fos_pairs[0].messages[0].content
        assert "3 allocation projects" in fos_pairs[0].messages[1].content
        # Biophysics has only 1 project → should NOT generate
        assert not any("Biophysics" in p.messages[0].content for p in fos_pairs)

    def test_allocations_by_institution(self):
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, allocations_data=ALLOCATIONS_DATA)

        inst_pairs = [p for p in pairs if p.id.startswith("cmp_alloc_inst")]
        # MIT has 3 projects → should generate
        assert len(inst_pairs) == 1
        assert "MIT" in inst_pairs[0].messages[0].content
        # Stanford has only 1 project → should NOT generate
        assert not any("Stanford" in p.messages[0].content for p in inst_pairs)

    def test_allocations_by_resource(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, {}, allocations_data=ALLOCATIONS_DATA)

        res_pairs = [p for p in pairs if p.id.startswith("cmp_alloc_res")]
        # "Delta GPU" is used by 3 projects → should generate
        assert len(res_pairs) == 1
        assert "Delta GPU" in res_pairs[0].messages[0].content

    def test_min_group_size_large(self):
        """Groups of size < 3 should be skipped for large domains."""
        small_data = {
            "TG-1": {"name": "P1", "fos": "Rare Field", "institution": "X"},
            "TG-2": {"name": "P2", "fos": "Rare Field", "institution": "Y"},
        }
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, allocations_data=small_data)
        # 2 < MIN_GROUP_SIZE_LARGE (3) → no pairs
        assert len(pairs) == 0

    def test_allocations_count_sample_format(self):
        """Large groups should use count+sample format."""
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, allocations_data=ALLOCATIONS_DATA)
        fos_pairs = [p for p in pairs if "cmp_alloc_fos" in p.id]
        assert len(fos_pairs) == 1
        answer = fos_pairs[0].messages[1].content
        # Should list all 3 (under max_sample=5) with colon separator
        assert ":" in answer
        # Should have citations for all listed items
        assert answer.count("<<SRC:allocations:") == 3


class TestNSFAwardsComparisons:
    """Test NSF awards comparison generation (new)."""

    def test_nsf_by_program(self):
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, nsf_awards_data=NSF_AWARDS_DATA)

        prog_pairs = [p for p in pairs if p.id.startswith("cmp_nsf_prog")]
        # OAC has 3 awards → should generate
        assert len(prog_pairs) == 1
        assert "OAC" in prog_pairs[0].messages[0].content
        assert "3 NSF awards" in prog_pairs[0].messages[1].content
        # PHY has only 1 award → should NOT generate
        assert not any("PHY" in p.messages[0].content for p in prog_pairs)

    def test_nsf_by_institution(self):
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, nsf_awards_data=NSF_AWARDS_DATA)

        inst_pairs = [p for p in pairs if p.id.startswith("cmp_nsf_inst")]
        # MIT has 2 awards, Stanford 1, Caltech 1 → none ≥ 3 → no pairs... wait
        # Actually: MIN_GROUP_SIZE_LARGE = 3, MIT has 2 → should NOT generate
        assert len(inst_pairs) == 0

    def test_nsf_by_institution_with_enough(self):
        """With 3+ awards at same institution, should generate."""
        data = {
            **NSF_AWARDS_DATA,
            "3333333": {
                "name": "More AI",
                "award_number": "3333333",
                "pi": "Dr. Z",
                "institution": "MIT",
                "primary_program": "CISE",
            },
        }
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, nsf_awards_data=data)
        inst_pairs = [p for p in pairs if p.id.startswith("cmp_nsf_inst")]
        assert len(inst_pairs) == 1
        assert "MIT" in inst_pairs[0].messages[0].content

    def test_nsf_pairs_have_correct_domain(self):
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, nsf_awards_data=NSF_AWARDS_DATA)
        for pair in pairs:
            if pair.id.startswith("cmp_nsf_"):
                assert pair.domain == "nsf-awards"


class TestAffinityGroupsComparisons:
    """Test affinity groups comparison generation (new)."""

    def test_affinity_by_category(self):
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, affinity_groups_data=AFFINITY_GROUPS_DATA)

        cat_pairs = [p for p in pairs if p.id.startswith("cmp_ag_cat")]
        # Technology has 3 groups → should generate (≥ MIN_GROUP_SIZE_SMALL=2)
        assert len(cat_pairs) == 1
        assert "Technology" in cat_pairs[0].messages[0].content
        # All 3 Technology groups should be listed
        answer = cat_pairs[0].messages[1].content
        assert "GPU Computing" in answer
        assert "Machine Learning" in answer
        assert "Cloud Computing" in answer
        # Community has only 1 → should NOT generate
        assert not any("Community" in p.messages[0].content for p in cat_pairs)

    def test_affinity_lists_all_members(self):
        """Small-domain comparison should list all group names (not count+sample)."""
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, affinity_groups_data=AFFINITY_GROUPS_DATA)
        cat_pairs = [p for p in pairs if p.id.startswith("cmp_ag_cat")]
        assert len(cat_pairs) == 1
        answer = cat_pairs[0].messages[1].content
        # Should have citations for all groups
        assert answer.count("<<SRC:affinity-groups:") == 3

    def test_affinity_pairs_have_correct_domain(self):
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, affinity_groups_data=AFFINITY_GROUPS_DATA)
        for pair in pairs:
            if pair.id.startswith("cmp_ag_"):
                assert pair.domain == "affinity-groups"


class TestCountSampleFormat:
    """Test the count+sample answer formatter."""

    def test_large_group_uses_sample(self):
        """Groups larger than max_sample should show 'including' with subset."""
        # Create 7 projects in same FOS → should cap sample at 5
        data = {}
        for i in range(7):
            pid = f"TG-{i:07d}"
            data[pid] = {
                "name": f"Project {i}",
                "fos": "Physics",
                "institution": f"Inst-{i}",
            }

        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, allocations_data=data)
        fos_pairs = [p for p in pairs if "cmp_alloc_fos" in p.id]
        assert len(fos_pairs) == 1
        answer = fos_pairs[0].messages[1].content
        assert "7 allocation projects" in answer
        assert "including" in answer  # Large group uses "including"
        # Should cite only 5 (max_sample)
        assert answer.count("<<SRC:allocations:") == 5

    def test_small_group_uses_colon(self):
        """Groups at or below max_sample should list all with colon separator."""
        data = {
            "TG-1": {"name": "A", "fos": "Math", "institution": "X"},
            "TG-2": {"name": "B", "fos": "Math", "institution": "Y"},
            "TG-3": {"name": "C", "fos": "Math", "institution": "Z"},
        }
        gen = ComparisonGenerator()
        pairs = gen.generate({}, {}, allocations_data=data)
        fos_pairs = [p for p in pairs if "cmp_alloc_fos" in p.id]
        assert len(fos_pairs) == 1
        answer = fos_pairs[0].messages[1].content
        assert ":" in answer  # Small group uses colon
        assert "including" not in answer


class TestBackwardsCompatibility:
    """Test that existing 2-arg callers still work."""

    def test_generate_with_two_args(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(COMPUTE_DATA, SOFTWARE_DATA)
        # Should still work — generates compute + software comparisons
        assert len(pairs) > 0
        assert all(p.metadata.granularity == "comparison" for p in pairs)

    def test_none_domains_skipped(self):
        gen = ComparisonGenerator()
        pairs = gen.generate(
            COMPUTE_DATA,
            SOFTWARE_DATA,
            allocations_data=None,
            nsf_awards_data=None,
            affinity_groups_data=None,
        )
        # Should only have compute + software pairs, no allocation/nsf/affinity
        for pair in pairs:
            assert not pair.id.startswith("cmp_alloc_")
            assert not pair.id.startswith("cmp_nsf_")
            assert not pair.id.startswith("cmp_ag_")


class TestGranularityModel:
    """Test that the granularity field works correctly on QAPair."""

    def test_default_granularity_is_comprehensive(self):
        from access_qa_extraction.models import QAPair

        pair = QAPair.create(
            id="test_1",
            question="Test?",
            answer="Test. <<SRC:test:1>>",
            source_ref="mcp://test/1",
            domain="test",
        )
        assert pair.metadata.granularity == "comprehensive"

    def test_comparison_granularity(self):
        from access_qa_extraction.models import QAPair

        pair = QAPair.create(
            id="test_3",
            question="Test?",
            answer="Test. <<SRC:test:3>>",
            source_ref="mcp://test/3",
            domain="test",
            granularity="comparison",
        )
        assert pair.metadata.granularity == "comparison"
