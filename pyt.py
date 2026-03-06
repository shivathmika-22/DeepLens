def solve():
    """
    Reads array input from standard input lines and calculates the maximum number 
    of partitions that meet the difference >= 2 criterion.
    """
    import sys
    
    # 1. Input N (length of the array)
    try:
        n_line = sys.stdin.readline().strip()
        if not n_line:
            return
        n = int(n_line)
        if n <= 0:
            return

        # 2. Input array elements
        arr_line = sys.stdin.readline().strip()
        if not arr_line:
            return

        arr = list(map(int, arr_line.split()))

    except (EOFError, ValueError):
        return
        
    # --- Core Logic ---
    if not arr:
        print(0)
        return

    # 1. Get unique elements (remove consecutive duplicates)
    unique_arr = []
    unique_arr.append(arr[0])
    for x in arr[1:]:
        if x != unique_arr[-1]:
            unique_arr.append(x)

    # 2. Apply greedy strategy
    count = 1  # First unique element always counted
    last_kept = unique_arr[0]
    
    for i in range(1, len(unique_arr)):
        current_val = unique_arr[i]
        if current_val - last_kept >= 2:
            count += 1
            last_kept = current_val
            
    # 3. Output the result
    print(count)


if __name__ == "__main__":
    solve()
