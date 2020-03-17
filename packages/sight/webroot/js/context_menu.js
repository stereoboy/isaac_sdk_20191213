/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Create a context menu that can be attached to any object of the DOM.
class ContextMenu {
  constructor(menu) {
    this.menu_ = document.createElement("div");
    this.items_ = [];
    this.menu_.className += " sight-context-menu";
    for (let i in menu) {
      this.addItem(menu[i]);
    }
    const that = this;
    // Hide the menu after a click
    window.addEventListener("click", function(evt) {
      // Firefox fire an event click after the mouseup
      if (evt.button != 2 || that.menu_.parentNode === null ||
          evt.target.id != that.menu_.parentNode.id) {
        that.menu_.style.display = "";
      }
    }, false);
    // Disable context menu on the menu
    this.menu_.oncontextmenu = function(evt) {
      evt.preventDefault();
      return false;
    };
  }

  // Add a new item to the menu:
  // item: {
  //   text: String to be displayed
  //   callback: callback function for a clic or slider update
  //   enable: default true, wether or not we can interract with it
  //   slider: Information about the slider {min = 0, max = 1, value = 0.5},
  //   submenu: List of items to be displayed on a submenu (works recursively)
  // }
  addItem(item) {
    let div = document.createElement("div");
    div.appendChild(document.createTextNode(""));
    div.callback = item.callback;
    div.updateTitle = function (text) {
      div.firstChild.nodeValue = text;
    }
    if (item.text) div.updateTitle(item.text);
    if (item.slider) {
      item.enable = false;
      if (!item.slider.min) item.slider.min = 0.0;
      if (!item.slider.max) item.slider.max = 1.0;
      if (!item.slider.value) item.slider.value =  0.5 * (item.slider.max + item.slider.min);
      let slider = document.createElement('input');
      let slider_out = document.createElement('output');
      slider_out.style.display = "inline";
      slider.type = "range";
      slider.className = "slider-context-menu";
      slider.min = item.slider.min;
      slider.max = item.slider.max;
      slider.step = "any";
      slider.value = item.slider.value;
      slider_out.value = item.slider.value;
      div.appendChild(slider);
      div.appendChild(slider_out);
      slider.oninput = function() {
        // Round to two decimals
        this.value = ((100.0 * this.value) | 0) / 100;
        slider_out.value = this.value;
        item.callback(this.value);
      };
      console.log(slider);
    }
    div.onclick = function() {
      if (this.enable && this.callback) this.callback();
    };
    this.menu_.appendChild(div);
    this.items_.push(div);
    if (item.enable !== false) {
      this.enableItem(this.items_.length - 1);
    } else {
      this.disableItem(this.items_.length - 1);
    }
    if (item.submenu) {
      div.className += " sight-context-menu-with-submenu";
      let submenu = new ContextMenu(item.submenu);
      div.appendChild(submenu.menu_);
      div.onmouseover = function() {
        const x = div.parentNode.offsetLeft + div.parentNode.offsetWidth - 5;
        const y = div.offsetTop + div.parentNode.offsetTop;
        submenu.menu_.style.left = x + "px";
        submenu.menu_.style.top = y + "px";
        submenu.menu_.style.display = "block";
      }
      div.onmouseleave = function() {
        submenu.menu_.style.display = "";
      }
    }
  }

  // Remove every element of the context menu
  clear() {
    for (let i = 0; i < this.items_.length; i++) {
      this.menu_.removeChild(this.items_[i]);
    }
    this.items_ = [];
  }

  // Remove an item from the menu
  removeItem(id) {
    this.menu_.removeChild(this.items_[id]);
    this.items_.splice(id, 1);
  }

  // Set an item as disable (can't click on it anymire)
  disableItem(id) {
    this.items_[id].false = false;
    this.items_[id].className = "sight-context-menu-item-disable";
  }

  // Enable an item (click on it will trigger its callback)
  enableItem(id) {
    this.items_[id].enable = true;
    this.items_[id].className = "sight-context-menu-item";
  }

  // Show the menu at a given position
  show(x, y) {
    // Hide other menu
    this.menu_.click();
    // Render the current menu at the position of the click
    this.menu_.style.left = x + "px";
    this.menu_.style.top = y + "px";
    this.menu_.style.display = "block";
  }

  // Attach the menu to an object.
  attachTo(parent) {
    if (parent.id == "") parent.id = '_' + Math.random().toString(36).substr(2, 9);
    // Append to the body as some object cannot contain the menu such as a canvas
    document.body.appendChild(this.menu_);

    const that = this;
    // Disable the default context menu
    parent.addEventListener("contextmenu", function(evt) {
      that.show(evt.clientX-1, evt.clientY-1);
      // Prevent the default context menu to show up
      evt.preventDefault();
    }, false);
  }
}