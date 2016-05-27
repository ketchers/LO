var count = 0;

btn = document.getElementById("btn");

btn.addEventListener('click', function() {
    ent = document.getElementById("entry");
    submit(ent)
});

function submit(ent) {
    var txt = ent.value // .replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br/>');
    var dv = document.getElementById("posts");
    var par = document.createElement("P");
    // Create a <p> element
    // var t = document.createTextNode("Post " + count + ": " + txt);
    // Create a text node
    // par.appendChild(t);
    count = count + 1;
    par.setAttribute('id', 'par' + count);
    par.innerHTML = txt;
    par.setAttribute('class', 'posts');
    // Append the text to <p>
    dv.appendChild(par);
    MathJax.Hub.Queue(["Typeset", MathJax.Hub, 'par' + count]);
    ent.value = "";
    ent.placeholder = "Output Here.";
}