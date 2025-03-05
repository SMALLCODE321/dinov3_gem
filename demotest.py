import tkinter as tk

# 初始化状态
current_state = "State 1"

def switch_to_state1():
    """切换到状态1"""
    global current_state
    if current_state != "State 1":
        current_state = "State 1"
        label.config(text="Current State: State 1")
        button1.config(text="Switch to State 1")
        button2.config(text="Switch to State 2")
        do_state_1_things()

def switch_to_state2():
    """切换到状态2"""
    global current_state
    if current_state != "State 2":
        current_state = "State 2"
        label.config(text="Current State: State 2")
        button1.config(text="Switch to State 1")
        button2.config(text="Switch to State 2")
        do_state_2_things()

def do_state_1_things():
    """执行状态1特有的操作"""
    print("Doing things for State 1")

def do_state_2_things():
    """执行状态2特有的操作"""
    print("Doing things for State 2")

# 创建主窗口
root = tk.Tk()
root.title("State Switcher")

# 创建标签用于显示当前状态
label = tk.Label(root, text="Current State: State 1")
label.pack()

# 创建按钮
button1 = tk.Button(root, text="Switch to State 1", command=switch_to_state1)
button1.pack()

button2 = tk.Button(root, text="Switch to State 2", command=switch_to_state2)
button2.pack()

# 开始主事件循环
root.mainloop()