import ebmlite


def iter_hierarchy(ebmldoc, nocache=False):
    """
    Iterate depth-first over all elements, yielding for each element its
    hierarchical path from the root document.

    :param nocache: whether to skip parser caching (defaults to False)
    :send: whether the current element's descendants should be skipped;
        defaults to None -> False
    :send type: bool
    :yield: the hierarchical path to an element from the document root for
        every element in the document
    :yield type: list
    """
    should_skip = yield [ebmldoc]
    if should_skip:
        return

    try:
        iter_subelements = ebmldoc.__iter__(nocache=nocache)
    except AttributeError:
        return

    for subelement in iter_subelements:
        iter_subhierarchy = iter_hierarchy(subelement, nocache=nocache)

        should_skip = None  # must start an iterator by sending `None`
        try:
            while True:
                should_skip = yield [ebmldoc] + iter_subhierarchy.send(should_skip)
        except StopIteration:
            continue


def filter_elements_by_name(doc, elem_names):
    """Filter hierarchical element paths by element names."""
    elem_names = [doc.name] + elem_names
    iter_elem_hierarchy = iter_hierarchy(doc)
    should_skip = None

    try:
        while True:
            element_path = iter_elem_hierarchy.send(should_skip)
            should_skip = element_path[-1].name != elem_names[len(element_path) - 1]
            if not should_skip and len(element_path) == len(elem_names):
                yield element_path
    except StopIteration:
        pass


def iter_config_attrs(filename, config_id, element_name):
    """Determine from file whether to perform segment processing."""
    with open(filename, "rb") as file:
        doc = ebmlite.loadSchema("mide_ide.xml").load(file)
        for elem_hierarchy in filter_elements_by_name(
            doc,
            [
                "RecorderConfigurationList",
                "RecorderConfigurationItem",
                "ConfigID",
            ],
        ):
            if elem_hierarchy[-1].value != config_id:
                continue
            for elem in elem_hierarchy[-2]:
                if elem.name == element_name:
                    yield elem.value
