import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import tracemalloc
from collections import deque
import heapq
from mazelib import Maze
from mazelib.generate.Prims import Prims
import pandas as pd

def print_maze_with_path(maze, path):
    thin = " "
    maze_copy = [row[:] for row in maze]
    for r, c in path:
        maze_copy[r][c] = "-"
    lines = []
    for row in maze_copy:
        lines.append(thin.join(str(cell) for cell in row))
    return "\n".join(lines)

def build_mdp(maze, goal):
    maze_rows = len(maze)
    maze_cols = len(maze[0])

    ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    gamma = 0.99
    step_reward = -1
    goal_reward = 0

    states = [(r, c) for r in range(maze_rows) for c in range(maze_cols) if maze[r][c] == 0]
    def is_terminal(s):
        return s == goal

    def next_state_reward(s, a):
        if is_terminal(s):
            return s, goal_reward
        r, c = s
        dr, dc = a
        nr, nc = r + dr, c + dc
        if not (0 <= nr < maze_rows and 0 <= nc < maze_cols):
            return s, step_reward
        if maze[nr][nc] == 1:
            return s, step_reward
        if (nr, nc) == goal:
            return (nr, nc), goal_reward
        return (nr, nc), step_reward

    return states, ACTIONS, gamma, is_terminal, next_state_reward

def policy_iteration(states, ACTIONS, gamma, is_terminal, next_state_reward, max_iters=1000):

    tracemalloc.start()

    policy = {s: random.choice(ACTIONS) if not is_terminal(s) else None for s in states}
    V = {s: 0 for s in states}

    t0 = time.perf_counter()
    iters = 0
    stable = False
    nodes_visited = 0

    while not stable and iters < max_iters:
        while True:
            delta = 0
            for s in states:
                if is_terminal(s):
                    continue
                old_v = V[s]
                a = policy[s]
                ns, r = next_state_reward(s, a)
                V[s] = r + gamma * V[ns]
                delta = max(delta, abs(old_v - V[s]))
                nodes_visited += 1  
            if delta < 1e-4:
                break
        stable = True
        for s in states:
            if is_terminal(s):
                continue
            old_a = policy[s]
            best_a = max(
                ACTIONS,
                key=lambda a: next_state_reward(s, a)[1] + gamma * V[next_state_reward(s, a)[0]]
            )
            policy[s] = best_a
            if best_a != old_a:
                stable = False
            nodes_visited += 1  

        iters += 1

    runtime = time.perf_counter() - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return V, policy, iters, runtime, nodes_visited, current_mem, peak_mem

def extract_path(policy, start, goal, next_state_reward, max_steps=10000):
    path = []
    s = start
    steps = 0
    while s != goal and steps < max_steps:
        path.append(s)
        a = policy.get(s, None)
        if a is None:
            break
        ns, _ = next_state_reward(s, a)
        if ns == s:
            break
        s = ns
        steps += 1
    if s == goal:
        path.append(goal)
    return path

def bfs_solve(maze, start, goal):
    rows, cols = len(maze), len(maze[0])

    tracemalloc.start()

    queue = deque([start])
    visited = set([start])
    parent = {start: None}

    nodes_expanded = 0
    t0 = time.perf_counter()

    while queue:
        current = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            break

        r, c = current
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = current
                    queue.append((nr, nc))

    runtime = time.perf_counter() - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = []
    if goal in parent:
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    return {
        "path": path,
        "path_length": len(path),
        "nodes_expanded": nodes_expanded,
        "runtime": runtime,
        "current_memory_bytes": current_mem,
        "peak_memory_bytes": peak_mem,
        "found": goal in parent
    }

def dfs_solve(maze, start, goal):
    rows, cols = len(maze), len(maze[0])

    tracemalloc.start()

    stack = [start]
    visited = set([start])
    parent = {start: None}

    nodes_expanded = 0
    t0 = time.perf_counter()

    while stack:
        current = stack.pop()
        nodes_expanded += 1

        if current == goal:
            break

        r, c = current
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = current
                    stack.append((nr, nc))

    runtime = time.perf_counter() - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = []
    if goal in parent:
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    return {
        "path": path,
        "path_length": len(path),
        "nodes_expanded": nodes_expanded,
        "runtime": runtime,
        "current_memory_bytes": current_mem,
        "peak_memory_bytes": peak_mem,
        "found": goal in parent
    }

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_manhattan_solve(maze, start, goal):
    rows, cols = len(maze), len(maze[0])

    tracemalloc.start()

    open_set = []
    heapq.heappush(open_set, (0, start))

    g = {start: 0}
    parent = {start: None}
    visited = set()

    nodes_expanded = 0
    t0 = time.perf_counter()

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal:
            break

        if current in visited:
            continue
        visited.add(current)

        r, c = current
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                new_cost = g[current] + 1

                if (nr, nc) not in g or new_cost < g[(nr, nc)]:
                    g[(nr, nc)] = new_cost
                    f = new_cost + manhattan((nr, nc), goal)
                    parent[(nr, nc)] = current
                    heapq.heappush(open_set, (f, (nr, nc)))

    runtime = time.perf_counter() - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = []
    if goal in parent:
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    return {
        "path": path,
        "path_length": len(path),
        "nodes_expanded": nodes_expanded,
        "runtime": runtime,
        "current_memory_bytes": current_mem,
        "peak_memory_bytes": peak_mem,
        "found": goal in parent
    }

def euclidean(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def astar_euclidean_solve(maze, start, goal):
    rows, cols = len(maze), len(maze[0])

    tracemalloc.start()

    open_set = []
    heapq.heappush(open_set, (0, start))

    g = {start: 0}
    parent = {start: None}
    visited = set()

    nodes_expanded = 0
    t0 = time.perf_counter()

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal:
            break

        if current in visited:
            continue
        visited.add(current)

        r, c = current
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                new_cost = g[current] + 1

                if (nr, nc) not in g or new_cost < g[(nr, nc)]:
                    g[(nr, nc)] = new_cost
                    f = new_cost + euclidean((nr, nc), goal)
                    parent[(nr, nc)] = current
                    heapq.heappush(open_set, (f, (nr, nc)))

    runtime = time.perf_counter() - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = []
    if goal in parent:
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    return {
        "path": path,
        "path_length": len(path),
        "nodes_expanded": nodes_expanded,
        "runtime": runtime,
        "current_memory_bytes": current_mem,
        "peak_memory_bytes": peak_mem,
        "found": goal in parent
    }

def value_iteration(states, ACTIONS, gamma, is_terminal, next_state_reward, theta=1e-4, max_iters=1000):

    tracemalloc.start()

    V = {s: 0 for s in states}
    iters = 0
    t0 = time.perf_counter()
    nodes_visited = 0

    while True:
        delta = 0
        for s in states:
            if is_terminal(s):
                continue
            old_v = V[s]
            V[s] = max(
                next_state_reward(s, a)[1] + gamma * V[next_state_reward(s, a)[0]]
                for a in ACTIONS
            )
            delta = max(delta, abs(old_v - V[s]))
            nodes_visited += 1  
        iters += 1
        if delta < theta or iters >= max_iters:
            break

    runtime = time.perf_counter() - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    policy = {}
    for s in states:
        if is_terminal(s):
            policy[s] = None
            continue
        best_a = max(
            ACTIONS,
            key=lambda a: next_state_reward(s, a)[1] + gamma * V[next_state_reward(s, a)[0]]
        )
        policy[s] = best_a

    return V, policy, iters, runtime, nodes_visited, current_mem, peak_mem

st.title("Maze Solver – Multi‑Algorithm Pathfinding (Streamlit GUI)")

with st.sidebar:
    st.header("Maze Configuration")
    rows = st.number_input("Rows", min_value=5, max_value=101, value=20, key="rows_input")
    cols = st.number_input("Columns", min_value=5, max_value=101, value=20, key="cols_input")
    maze_type = st.selectbox(
        "Maze Type",
        ["Single Path", "Multiple Path"]
    )

    generate_button = st.button("Generate Maze")
    algo = st.selectbox(
        "Choose Algorithm",
        ["BFS", "DFS", "A* Manhattan", "A* Euclidean", "Value Iteration", "Policy Iteration"]
    )

    run_button = st.button("Run Algorithm")
    selected_algorithms = st.multiselect(
        "Select Algorithms to Run",
        ["BFS", "DFS", "A* Manhattan", "A* Euclidean", "Value Iteration", "Policy Iteration"],
        default=[]
    )

    run_selected_button = st.button("Run Selected Algorithms")

if "maze" not in st.session_state:
    st.session_state.maze = None
if "path" not in st.session_state:
    st.session_state.path = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if generate_button:
    random.seed(345)
    np.random.seed(345)
    m = Maze()
    m.generator = Prims(rows, cols)
    m.generate()
    maze = m.grid.tolist()
    if maze_type == "Multiple Path":
        for _ in range((rows * cols) // 10):
            r = random.randint(1, rows - 2)
            c = random.randint(1, cols - 2)
            maze[r][c] = 0
    start = (1, 1)
    goal = (rows - 2, cols - 2)
    maze[start[0]][start[1]] = 0
    maze[goal[0]][goal[1]] = 0
    st.session_state.maze = maze
    st.session_state.path = None
    st.session_state.metrics = None
    st.success("Maze generated successfully.")
if run_button:
    if st.session_state.maze is None:
        st.error("Generate a maze first.")
    else:
        maze = st.session_state.maze
        rows = len(maze)
        cols = len(maze[0])

        start = (1, 1)
        goal = (rows - 2, cols - 2)

        if algo == "Policy Iteration":
            states, ACTIONS, gamma, is_terminal, next_state_reward = build_mdp(maze, goal)
            V, policy, iters, runtime, nodes_visited, current_mem, peak_mem = policy_iteration(
                states, ACTIONS, gamma, is_terminal, next_state_reward
            )
            path = extract_path(policy, start, goal, next_state_reward)

            st.session_state.path = path
            st.session_state.metrics = {
                "Algorithm": "Policy Iteration",
                "Iterations": iters,
                "Runtime (s)": runtime,
                "Path Length": len(path),
                "Visited Nodes": nodes_visited,
                "Peak Memory (bytes)": peak_mem,
            }

        elif algo == "BFS":
            result = bfs_solve(maze, start, goal)

            st.session_state.path = result["path"]
            st.session_state.metrics = {
                "Algorithm": "BFS",
                "Runtime (s)": result["runtime"],
                "Path Length": result["path_length"],
                "Visited Nodes": result["nodes_expanded"],
                "Peak Memory (bytes)": result["peak_memory_bytes"],
            }

        elif algo == "DFS":
            result = dfs_solve(maze, start, goal)

            st.session_state.path = result["path"]
            st.session_state.metrics = {
                "Algorithm": "DFS",
                "Runtime (s)": result["runtime"],
                "Path Length": result["path_length"],
                "Visited Nodes": result["nodes_expanded"],
                "Peak Memory (bytes)": result["peak_memory_bytes"],
            }

        elif algo == "A* Manhattan":
            result = astar_manhattan_solve(maze, start, goal)

            st.session_state.path = result["path"]
            st.session_state.metrics = {
                "Algorithm": "A* Manhattan",
                "Runtime (s)": result["runtime"],
                "Path Length": result["path_length"],
                "Visited Nodes": result["nodes_expanded"],
                "Peak Memory (bytes)": result["peak_memory_bytes"],
            }

        elif algo == "A* Euclidean":
            result = astar_euclidean_solve(maze, start, goal)

            st.session_state.path = result["path"]
            st.session_state.metrics = {
                "Algorithm": "A* Euclidean",
                "Runtime (s)": result["runtime"],
                "Path Length": result["path_length"],
                "Visited Nodes": result["nodes_expanded"],
                "Peak Memory (bytes)": result["peak_memory_bytes"],
            }

        elif algo == "Value Iteration":
            states, ACTIONS, gamma, is_terminal, next_state_reward = build_mdp(maze, goal)
            V, policy, iters, runtime, nodes_visited, current_mem, peak_mem = value_iteration(
                states, ACTIONS, gamma, is_terminal, next_state_reward
            )
            path = extract_path(policy, start, goal, next_state_reward)

            st.session_state.path = path
            st.session_state.metrics = {
                "Algorithm": "Value Iteration",
                "Iterations": iters,
                "Runtime (s)": runtime,
                "Path Length": len(path),
                "Visited Nodes": nodes_visited,
                "Peak Memory (bytes)": peak_mem,
            }
if run_selected_button:
    if st.session_state.maze is None:
        st.error("Generate a maze first.")
    elif len(selected_algorithms) == 0:
        st.error("Select at least one algorithm.")
    else:
        maze = st.session_state.maze
        rows = len(maze)
        cols = len(maze[0])

        start = (1, 1)
        goal = (rows - 2, cols - 2)

        results = []
        paths = {}

        for algo_name in selected_algorithms:

            if algo_name == "BFS":
                result = bfs_solve(maze, start, goal)
                results.append([
                    "BFS",
                    result["runtime"],
                    result["path_length"],
                    result["nodes_expanded"],
                    result["peak_memory_bytes"]
                ])
                paths["BFS"] = result["path"]

            elif algo_name == "DFS":
                result = dfs_solve(maze, start, goal)
                results.append([
                    "DFS",
                    result["runtime"],
                    result["path_length"],
                    result["nodes_expanded"],
                    result["peak_memory_bytes"]
                ])
                paths["DFS"] = result["path"]

            elif algo_name == "A* Manhattan":
                result = astar_manhattan_solve(maze, start, goal)
                results.append([
                    "A* Manhattan",
                    result["runtime"],
                    result["path_length"],
                    result["nodes_expanded"],
                    result["peak_memory_bytes"]
                ])
                paths["A* Manhattan"] = result["path"]

            elif algo_name == "A* Euclidean":
                result = astar_euclidean_solve(maze, start, goal)
                results.append([
                    "A* Euclidean",
                    result["runtime"],
                    result["path_length"],
                    result["nodes_expanded"],
                    result["peak_memory_bytes"]
                ])
                paths["A* Euclidean"] = result["path"]

            elif algo_name == "Value Iteration":
                states, ACTIONS, gamma, is_terminal, next_state_reward = build_mdp(maze, goal)
                V, policy, iters, runtime, nodes_visited, current_mem, peak_mem = value_iteration(
                    states, ACTIONS, gamma, is_terminal, next_state_reward
                )
                vi_path = extract_path(policy, start, goal, next_state_reward)
                results.append([
                    "Value Iteration",
                    runtime,
                    len(vi_path),
                    nodes_visited,
                    peak_mem
                ])
                paths["Value Iteration"] = vi_path

            elif algo_name == "Policy Iteration":
                states, ACTIONS, gamma, is_terminal, next_state_reward = build_mdp(maze, goal)
                V, policy, iters, runtime, nodes_visited, current_mem, peak_mem = policy_iteration(
                    states, ACTIONS, gamma, is_terminal, next_state_reward
                )
                pi_path = extract_path(policy, start, goal, next_state_reward)
                results.append([
                    "Policy Iteration",
                    runtime,
                    len(pi_path),
                    nodes_visited,
                    peak_mem
                ])
                paths["Policy Iteration"] = pi_path
        df = pd.DataFrame(
            results,
            columns=["Algorithm", "Runtime (s)", "Path Length", "Visited Nodes", "Peak Memory"]
        )
        st.subheader("Comparison of Selected Algorithms")
        st.dataframe(df)
        st.subheader("Visual Comparison of Selected Algorithm Paths")
        fig, ax = plt.subplots(figsize=(6, 6))
        maze_img = np.array(maze)
        ax.imshow(maze_img, cmap="binary")
        ax.axis("off")
        colors = {
            "BFS": "red",
            "DFS": "blue",
            "A* Manhattan": "green",
            "A* Euclidean": "orange",
            "Value Iteration": "purple",
            "Policy Iteration": "cyan"
        }
        offsets = {
            "BFS": (0.0, 0.0),
            "DFS": (0.45, 0.45),
            "A* Manhattan": (-0.45, 0.45),
            "A* Euclidean": (0.45, -0.45),
            "Value Iteration": (-0.45, -0.45),
            "Policy Iteration": (0.0, 0.45),
        }
        for algo_name, algo_path in paths.items():
            if algo_path:
                dy, dx = offsets.get(algo_name, (0, 0))
                ys = [p[0] + dy for p in algo_path]
                xs = [p[1] + dx for p in algo_path]
                ax.plot(xs, ys, color=colors[algo_name], linewidth=2, label=algo_name)

        ax.scatter([1], [1], color="yellow", s=50)
        ax.scatter([cols - 2], [rows - 2], color="lime", s=50)

        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig)
if not run_selected_button:
    maze = st.session_state.maze
    path = st.session_state.path
    metrics = st.session_state.metrics

    if maze is not None:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("Metrics")
            if metrics:
                for k, v in metrics.items():
                    st.write(f"**{k}:** {v}")
            else:
                st.write("Run Algorithm to see metrics.")
        with col2:
            st.subheader("Maze Display")
            fig, ax = plt.subplots(figsize=(5, 5))
            maze_img = np.array(maze)
            ax.imshow(maze_img, cmap="binary")
            ax.axis("off")
            if path:
                ys = [p[0] for p in path]
                xs = [p[1] for p in path]
                ax.plot(xs, ys, color="red", linewidth=2)
                ax.scatter([1], [1], color="yellow", s=40)
                ax.scatter([cols - 2], [rows - 2], color="lime", s=40)
            st.pyplot(fig)
        if path:
            st.subheader("Text View of Maze with Path")
            st.code(print_maze_with_path(maze, path))

