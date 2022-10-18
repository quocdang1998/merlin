// add class name to a dd object with the value of corresponding dt given
function append_class_to_corresponding_search_element (searchText, className) {
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
function change_attribute () {
    append_class_to_corresponding_search_element ("Type", "attributetype");
}

// set id for the last item in breadcrumbs
function set_id_lastchild_in_breadcrumbs () {
    var breadcrumbs = document.getElementsByClassName("wy-breadcrumbs")[0];
    var second_to_last = breadcrumbs.children[breadcrumbs.children.length - 2];
    second_to_last.setAttribute("id", "currentpage-breadcrumbs");
}

// lighten pygment color
function lighten_pygment_colors () {
    var hls = document.getElementsByClassName("highlight");
    for (var i = 0; i < hls.length; i++) {
        var hl = hls[i];
        var pre_hl = hl.childNodes[0];
        var spans = pre_hl.getElementsByTagName("span");
        for (var j = 0; j < spans.length; j++) {
            var span_color = getComputedStyle(spans[j]).color;
            var rgb = span_color.replace(/[^\d,]/g, '').split(',').map(x => parseInt(x, 10));
            var hsl = rgb_to_hsl(rgb[0], rgb[1],  rgb[2]);
            hsl[2] = (hsl[2] < 0.8) ? 0.8 : hsl[2];
            rgb = hsl_to_rgb(hsl[0], hsl[1], hsl[2]);
            var rgb_str = "rgb(" + rgb.join(", ") + ")";
            spans[j].style.color = rgb_str;
        }
    }
}

// restore pygment color
function restore_pygment_colors () {
    var hls = document.getElementsByClassName("highlight");
    for (var i = 0; i < hls.length; i++) {
        var hl = hls[i];
        var pre_hl = hl.childNodes[0];
        var spans = pre_hl.getElementsByTagName("span");
        for (var j = 0; j < spans.length; j++) {
            spans[j].style.color = "";
        }
    }
}

// set color scheme
function set_color_scheme () {
    var color_scheme;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        color_scheme = localStorage.getItem("color-scheme") || "dark";
    } else {
        color_scheme = localStorage.getItem("color-scheme") || "light";
    }
    document.documentElement.setAttribute("color-scheme", color_scheme);
    localStorage.setItem("color-scheme", color_scheme);
    if (color_scheme === "dark") {
        lighten_pygment_colors();
    }
}

// add darkmode button
function add_darkmode_button () {
    // create button and set id
    var darkmode_button = document.createElement("button");
    darkmode_button.setAttribute("id", "darkmode-button");
    var header = document.getElementsByTagName("h1")[0];
    header.appendChild(darkmode_button);
    // load svg image to content
    darkmode_button = document.getElementById("darkmode-button");
    darkmode_button.style.userSelect = "none";
    var sun_img = document.createElement("img");
    sun_img.setAttribute("id", "sun-button");
    sun_img.setAttribute("class", "button-image");
    sun_img.setAttribute("src", "http://simpleicon.com/wp-content/uploads/sun.svg");
    sun_img.setAttribute("title", "Light mode");
    darkmode_button.appendChild(sun_img);
    var moon_img = document.createElement("img");
    moon_img.setAttribute("id", "moon-button");
    moon_img.setAttribute("class", "button-image");
    moon_img.setAttribute("src", "http://simpleicon.com/wp-content/uploads/sun_1.svg");
    moon_img.setAttribute("title", "Dark mode");
    darkmode_button.appendChild(moon_img);
    // set onclick function
    darkmode_button.onclick = () => {
        const color_scheme = document.documentElement.getAttribute("color-scheme");
        const new_scheme = (color_scheme === "light") ? "dark" : "light";
        document.documentElement.setAttribute("color-scheme", new_scheme);
        if (new_scheme === "dark") {
            lighten_pygment_colors();
        } else {
            restore_pygment_colors();
        }
        localStorage.setItem("color-scheme", new_scheme);
    };
}

function azura_main () {
    set_color_scheme();
    change_attribute();
    set_id_lastchild_in_breadcrumbs();
    add_darkmode_button();
}

azura_main();

