// add class name to a dd object with the value of corresponding dt given
function append_class_to_correspondant_search_elenemt (searchText, className) {
    var dts = document.getElementsByTagName("dt");
    var dds = document.getElementsByTagName("dd");
    for (var i = 0; i < dts.length; i++) {
      if (dts[i].textContent == searchText) {
        var cor_dd = dds[i];
        cor_dd.classList.add(className);
      }
    }
}

// append class to "Return type" and "Type"
function change_return_and_attribute () {
    append_class_to_correspondant_search_elenemt ("Return type", "return");
    append_class_to_correspondant_search_elenemt ("Type", "attributetype");
}

// set id for the last item in breadcrumbs
function set_id_lastchild_in_breadcrumbs () {
    var breadcrumbs = document.getElementsByClassName("wy-breadcrumbs")[0];
    var second_to_last = breadcrumbs.children[breadcrumbs.children.length - 2];
    second_to_last.setAttribute("id", "currentpage-breadcrumbs");
}

function azura_main () {
    change_return_and_attribute();
    set_id_lastchild_in_breadcrumbs();
}

azura_main()