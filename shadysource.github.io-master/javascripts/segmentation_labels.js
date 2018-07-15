/**************************************************************
 * This code was adapted from a tutorial by William Malone
 *  explaining how to make a javascript drawing app.
 * I would like to thank the internet for helping me do this.
 * The purpose of this project is to create a javascript app
 *  to speed up the labeling of picture data.
 **************************************************************/
var context;

var clickX = new Array();
var clickY = new Array();
var clickColor = new Array();
var clickDrag = new Array();
var idx = 0;

var paint;

//colors
var PersonRed = "#d10c0c";
var BicycleGreen = "#028c0e";
var MotorYellow = "#e6f727";
var CarBrown = "#441a04";
var BusGreen = "#009140";
var TrafficlightOrange = "#ff6614";
var BusStopBlk = "#0a0807";
var PotholeYellow = "#fff100";
var BenchGreen = "#00873c";
var ChairBlue = "#0097db";
var DogRed = "#e60012";
var CatYellow = "#f2e55c";
var TreeGreen = "#779438";
var DiningtableOrange = "#e8ac51";
var SinkRed = "#c7000b";
var ToiletBlue = "#54c3f1";
var DoorBlue = "#6c9bd2";

var curColor = PersonRed;
var image;
var curImgURL = "";
var imageURLFile = "https://raw.githubusercontent.com/warren30815/test/master/url.txt";
var imageURLs = new Array();
var imageSet = false;

var tmpLabel;
var labels = new Array();
labels.push("Number of labels: 0");

// var info = "";
// var infoSet = false;

$(window).on("load",function() {

document.getElementById("numLabels").innerHTML = labels[0];
context = document.getElementById('pictureCanvas').getContext("2d");

$.get(imageURLFile,function(data){
    imageURLs = data.split("\n");
    newImage();
});

// $.getJSON('https://freegeoip.net/json/?callback=?', function(data) {
//     info = JSON.stringify(data);
//     info = incriment(info);
//     infoSet = true;
// });

function enableDrawing(){
    //on mouse click in canvas
    $("#pictureCanvas").mousedown(function(e){
        var mouseX = e.pageX - this.offsetLeft;
        var mouseY = e.pageY - this.offsetTop;  
        paint = true;
        addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
        redraw(context);
        return false;
        }
    );

    //on mouse movement in canvas
    $("#pictureCanvas").mousemove(function(e){
        if(paint){
            addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
            redraw(context);
            return false;
        }
    });

    //mouse unclick action
    $("#pictureCanvas").mouseup(function(e){
        paint = false;
        return false;
    });

    //mouse leaves the canvas
    $("#pictureCanvas").mouseleave(function(e){
        paint = false;
        return false;
    });

    $("#clearButton").click(function(){
        context.drawImage(image, 0, 0, context.canvas.width, context.canvas.height);
        resetVars();
    });
}

$("#PersonBtn").click(function(){curColor = PersonRed});
$("#BicycleBtn").click(function(){curColor = BicycleGreen});
$("#MotorcycleBtn").click(function(){curColor = MotorYellow});
$("#CarBtn").click(function(){curColor = CarBrown});
$("#BusBtn").click(function(){curColor = BusGreen});
$("#TrafficlightBtn").click(function(){curColor = TrafficlightOrange});
$("#BusStopBtn").click(function(){curColor = BusStopBlk});
$("#PotholeBtn").click(function(){curColor = PotholeYellow});
$("#BenchBtn").click(function(){curColor = BenchGreen});
$("#ChairBtn").click(function(){curColor = ChairBlue});
$("#DogBtn").click(function(){curColor = DogRed});
$("#CatBtn").click(function(){curColor = CatYellow});
$("#TreeBtn").click(function(){curColor = TreeGreen});
$("#DiningtableBtn").click(function(){curColor = DiningtableOrange});
$("#SinkBtn").click(function(){curColor = SinkRed});
$("#ToiletBtn").click(function(){curColor = ToiletBlue});
$("#DoorBtn").click(function(){curColor = DoorBlue});

$("#submitButton").click(function(){
    labels.push(getLabel());
    if(labels[labels.length-1]=="EMPTY")
        labels.pop();
    labels[0] = "Number of labels: " + (labels.length-1).toString();
    document.getElementById("numLabels").innerHTML = labels[0];
});

$("#unSubmitButton").click(function(){
    if (labels.length > 1)
        labels.pop();
    labels[0] = "Number of labels: " + (labels.length-1).toString();
    document.getElementById("numLabels").innerHTML = labels[0];
});

$("#newImageButton").click(function(){
    newImage();
    resetVars();
});

//cool, but unnecessary
/*$("#reSubmitButton").click(function(){
    if (tmpLabels.length > 0)
        labels.push(tmpLabels.pop());
    labels[0] = "Number of labels: " + (labels.length-1).toString();
    document.getElementById("numLabels").innerHTML = labels[0];
    resetVars();
});*/

$("#downloadButton").click(function(){
    var d = new Date();
    var filename = d.getFullYear().toString() + "y" + d.getMonth().toString() + "m" + d.getDate().toString() 
                + "d" + d.getHours().toString() + "h" + d.getMinutes().toString() + "m" + d.getSeconds().toString()
                + "s" + d.getMilliseconds().toString();
    var labelsString = "";
    for (i = 0; i < labels.length; i++)
        labelsString = labels[i];
    var blob =  new Blob([labelsString],{type: "text/plain;charset=utf-8"});
    if (labels.length > 1)
        saveAs(blob, filename);
    //cool, but not necesary
    //$("#abortButton").click(function(){filesaver.abort();});
    tmpLabels = new Array();
    labels = new Array();
    labels.push("Number of labels: 0");
    document.getElementById("numLabels").innerHTML = labels[0];
});

function addClick(x, y, dragging){
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
    clickColor.push(curColor);
}

function redraw(ctx){
    ctx.lineJoin = "round";
    ctx.lineWidth = 10;
    
    for(var i=idx; i < clickX.length; i++){		
        ctx.beginPath();
        if(clickDrag[i] && i){
                ctx.moveTo(clickX[i-1], clickY[i-1]);
            }else{
                ctx.moveTo(clickX[i]-1, clickY[i]);
            }
            ctx.lineTo(clickX[i], clickY[i]);
            ctx.closePath();
            ctx.strokeStyle = clickColor[i];
            ctx.stroke();
    }

    idx = clickX.length;
}

function newImage(){
    var urlIdx = Math.floor((Math.random() * imageURLs.length));
    var newImg = new Image();
    newImg.onload = function(){
        image = newImg;
        curImgURL = imageURLs[urlIdx];
        context.drawImage(image, 0, 0, context.canvas.width, context.canvas.height);
        if (imageSet === false){
            enableDrawing();
            imageSet = true;
        }
    }
    newImg.src = imageURLs[urlIdx];
    
}

function resetVars(){
    idx = 0;
    clickX = new Array();
    clickY = new Array();
    clickColor = new Array();
    clickDrag = new Array();
}

function incriment(str){
    inc = new String()
    for (var i = 0; i < str.length; i++)
        inc = inc + String.fromCharCode(str.charCodeAt(i) + 1);
    return inc;
}

function decriment(str){
    dec = new String()
    for (var i = 0; i < str.length; i++)
        dec = dec + String.fromCharCode(str.charCodeAt(i) - 1);
    return dec;
}

function getLabel(){
    var datacontext = createNewContext(context.canvas.width, context.canvas.height);
    //clear image and reset index, them redraw onto new canvas:
    idx = 0;
    redraw(datacontext);
    //resize (shrink) the image:
    var tmpImg = getCanvasImg(datacontext),
        newWidth = context.canvas.width*0.1,
        newHeight = context.canvas.height*0.1;
    datacontext.clearRect(0, 0, newWidth, newHeight)
    datacontext.drawImage(tmpImg, 0, 0, newWidth, newHeight);
    var imgData = datacontext.getImageData(0, 0, newWidth, newHeight);
    newWidth = newWidth*4;//this is the with of the img in datapoints
    var data = imgData.data;
    //create label:
    var noData = true;
    var label = curImgURL + "\n";
    for(var i=0; i<data.length; i+=4){
        var red = data[i];
        var green = data[i+1];
        var blue = data[i+2];
        var hex = rgbToHex(red, green, blue)
        if (hex === PersonRed){
            label = label + "Person";
            if(noData) noData = false;
        }
        else if (hex === BicycleGreen){
            label = label + "Bicycle";
            if(noData) noData = false;
        }
        else if (hex === MotorYellow){
            label = label + "Motor";
            if(noData) noData = false;
        }
        else if (hex === CarBrown){
            label = label + "Car";
            if(noData) noData = false;
        }
        else if(hex === BusGreen){
            label = label + "Bus";
            if(noData) noData = false;
        }
        else if (hex === TrafficlightOrange){
            label = label + "Trafficlight";
            if(noData) noData = false;
        }
        else if(hex === BusStopBlk){
            label = label + "Busstop";
            if(noData) noData = false;
        }
        else if(hex === PotholeYellow){
            label = label + "Pothole";
            if(noData) noData = false;
        }
        else if(hex === BenchGreen){
            label = label + "bench";
            if(noData) noData = false;
        }
        else if(hex === ChairBlue){
            label = label + "chair";
            if(noData) noData = false;
        }
        else if(hex === DogRed){
            label = label + "dog";
            if(noData) noData = false;
        }
        else if(hex === CatYellow){
            label = label + "cat";
            if(noData) noData = false;
        }
        else if(hex === TreeGreen){
            label = label + "tree";
            if(noData) noData = false;
        }
        else if(hex === DiningtableOrange){
            label = label + "diningtable";
            if(noData) noData = false;
        }
        else if(hex === SinkRed){
            label = label + "sink";
            if(noData) noData = false;
        }
        else if(hex === ToiletBlue){
            label = label + "toilet";
            if(noData) noData = false;
        }
        else if(hex === DoorBlue){
            label = label + "door";
            if(noData) noData = false;
        }
        else{
            label = label + "0";// no target
        }
        if (i%(newWidth) === 0 && i > 0){
            label = label + "\n";
        }
    }    

    if(noData)
        return "EMPTY";

    return "\n" + label + "\n";
}

function createNewContext(width, height) {
    var canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas.getContext("2d");
}

function getCanvasImg(ctxt){
    var canvasImage = new Image();
    canvasImage.src = ctxt.canvas.toDataURL();
    canvasImage.onload = function(){
        return canvasImage;
    //return canvasImage.onload
    };
    return canvasImage.onload();
}

function rgbToHex(R,G,B){
    return "#"+toHex(R)+toHex(G)+toHex(B)
}

function toHex(n) {
    n = parseInt(n,10);
    if (isNaN(n)) return "00";
    n = Math.max(0,Math.min(n,255));
    return "0123456789abcdef".charAt((n-n%16)/16)
        + "0123456789abcdef".charAt(n%16);
}
});

