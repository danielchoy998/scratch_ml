def test2():
    a = "11"

    a_size = len(a)
    
    sum = 0
    for i in range(a_size):
        if a[a_size - i - 1] == "0":
            num = 0
        else :
            num = 2**i
        sum += num
    
    while sum > 0:
        
    print(sum)

test2()