from streamlit import set_page_config, Page, navigation


def pages_configer():
    set_page_config(
        page_title="Knowledge Base",
        page_icon=":material/school:",
        layout="centered",
        initial_sidebar_state="expanded",
    )


def pages_layout():
    """ Set the streamlit pages layout """
    elements: dict[str, list[str]] = {
        "page": ["subpages/home.py", "subpages/cluster.py"],
        "title": ["Home", "Cluster Explorer"],
        "icon": [":material/home:", ":material/segment:"],
    }

    structure: dict[str, list[Page]] = {
        "Introduction": [
            Page(page=elements["page"][0], title=elements["title"][0], icon=elements["icon"][0]),
        ],
        "Examples": [
            Page(page=elements["page"][1], title=elements["title"][1], icon=elements["icon"][1]),
        ],
    }
    pg: Page = navigation(structure, position="sidebar", expanded=True)
    pg.run()
