def arr_prefix(pre_str, arr):
    return [pre_str+s for s in arr]

def arr_suffix(post_str, arr):
    return [s+post_str for s in arr]

def arr_str_zip(arr1, arr2):
    a = []
    for i in range(len(arr1)):
        a.append(arr1[i]+arr2[i])
    return a

def arr_str_zip_with(arr1,s, arr2):
    a = []
    for i in range(len(arr1)):
        a.append(arr1[i]+s+arr2[i])
    return a

def arr_to_sql_select_list(delim, arr):
    return '{delim}\n   '.format(delim=delim).join(arr)

