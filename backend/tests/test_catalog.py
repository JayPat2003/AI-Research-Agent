from app.agents.schemas import CatalogFilter
from app.pipeline.catalog import filter_catalog, load_catalog


def test_catalog_loads_and_filters_glp1():
    rows = load_catalog()
    assert len(rows) >= 10
    result = filter_catalog(CatalogFilter(drug_class="GLP-1", status="completed"))
    assert result["n"] >= 1
    assert all(r["drug_class"] == "GLP-1" for r in result["rows"])
