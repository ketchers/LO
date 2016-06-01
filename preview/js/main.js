btn = document.getElementById("btn");

btn.addEventListener('click', function () {
    ent = document.getElementById("entry");
    submit(ent)
});

function submit(ent) {
    var txt = ent.value
    var dv = document.getElementById("posts");
    dv.innerHTML = txt;
    MathJax.Hub.Queue(["Typeset", MathJax.Hub, dv]);
    // ent.value = "";
    // ent.placeholder = "Output Here.";
}