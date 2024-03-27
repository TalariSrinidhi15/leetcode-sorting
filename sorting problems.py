#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Relative Ranks
def findRelativeRanks(score):
    sorted_score = sorted(score, reverse=True)
    rank_dict = {sorted_score[i]: i + 1 for i in range(len(sorted_score))}
    
    medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    
    result = []
    for s in score:
        rank = rank_dict[s]
        if rank <= 3:
            result.append(medals[rank - 1])
        else:
            result.append(str(rank))
    
    return result

# Example usage:
score1 = [5,4,3,2,1]
print(findRelativeRanks(score1))  # Output: ["Gold Medal","Silver Medal","Bronze Medal","4","5"]

score2 = [10,3,8,9,4]
print(findRelativeRanks(score2))  # Output: ["Gold Medal","5","Bronze Medal","Silver Medal","4"]


# In[2]:


##Array partition
def arrayPairSum(nums):
    nums.sort()
    max_sum = 0
    for i in range(0, len(nums), 2):
        max_sum += nums[i]
    return max_sum

# Example usage:
nums1 = [1,4,3,2]
print(arrayPairSum(nums1))  # Output: 4

nums2 = [6,2,6,5,1,2]
print(arrayPairSum(nums2))  # Output: 9


# In[3]:


##Longest Harmonous subsequence
def findLHS(nums):
    frequency = {}
    max_length = 0
    
    # Count frequency of each number
    for num in nums:
        frequency[num] = frequency.get(num, 0) + 1
    
    # Check for harmonious subsequence
    for num in frequency:
        if num + 1 in frequency:
            max_length = max(max_length, frequency[num] + frequency[num + 1])
    
    return max_length

# Example usage:
nums1 = [1,3,2,2,5,2,3,7]
print(findLHS(nums1))  # Output: 5

nums2 = [1,2,3,4]
print(findLHS(nums2))  # Output: 2

nums3 = [1,1,1,1]
print(findLHS(nums3))  # Output: 0


# In[4]:


##fair candy swap
def fairCandySwap(aliceSizes, bobSizes):
    total_alice = sum(aliceSizes)
    total_bob = sum(bobSizes)
    
    difference = (total_alice - total_bob) // 2
    
    bobSet = set(bobSizes)
    
    for candy in aliceSizes:
        if candy - difference in bobSet:
            return [candy, candy - difference]

# Example usage:
aliceSizes1 = [1,1]
bobSizes1 = [2,2]
print(fairCandySwap(aliceSizes1, bobSizes1))  # Output: [1, 2]

aliceSizes2 = [1,2]
bobSizes2 = [2,3]
print(fairCandySwap(aliceSizes2, bobSizes2))  # Output: [1, 2]

aliceSizes3 = [2]
bobSizes3 = [1,3]
print(fairCandySwap(aliceSizes3, bobSizes3))  # Output: [2, 3]


# In[5]:


##sort array by parity
def sortArrayByParity(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        # Move left pointer to find an odd number
        while left < right and nums[left] % 2 == 0:
            left += 1
        
        # Move right pointer to find an even number
        while left < right and nums[right] % 2 != 0:
            right -= 1
        
        # Swap the elements at left and right pointers
        nums[left], nums[right] = nums[right], nums[left]
        
        # Move pointers towards each other
        left += 1
        right -= 1
    
    return nums

# Example usage:
nums1 = [3,1,2,4]
print(sortArrayByParity(nums1))  # Output: [4,2,1,3] (or any other valid output)

nums2 = [0]
print(sortArrayByParity(nums2))  # Output: [0]


# In[6]:


##matrix cells in distance order
def allCellsDistOrder(rows, cols, rCenter, cCenter):
    distances = []
    
    for r in range(rows):
        for c in range(cols):
            distances.append((r, c, abs(r - rCenter) + abs(c - cCenter)))
    
    distances.sort(key=lambda x: x[2])
    
    return [[r, c] for r, c, _ in distances]

# Example usage:
rows1, cols1, rCenter1, cCenter1 = 1, 2, 0, 0
print(allCellsDistOrder(rows1, cols1, rCenter1, cCenter1))  # Output: [[0,0],[0,1]]

rows2, cols2, rCenter2, cCenter2 = 2, 2, 0, 1
print(allCellsDistOrder(rows2, cols2, rCenter2, cCenter2))  # Output: [[0,1],[0,0],[1,1],[1,0]]

rows3, cols3, rCenter3, cCenter3 = 2, 3, 1, 2
print(allCellsDistOrder(rows3, cols3, rCenter3, cCenter3))  # Output: [[1,2],[0,2],[1,1],[0,1],[1,0],[0,0]]


# In[7]:


##Two city scheduling
def twoCitySchedCost(costs):
    # Sort the costs based on the difference between flying to city A and city B
    costs.sort(key=lambda x: x[0] - x[1])
    
    n = len(costs) // 2
    min_cost = 0
    
    # Send the first n people with the smallest differences to city A
    for i in range(n):
        min_cost += costs[i][0]
    
    # Send the rest to city B
    for i in range(n, len(costs)):
        min_cost += costs[i][1]
    
    return min_cost

# Example usage:
costs1 = [[10,20],[30,200],[400,50],[30,20]]
print(twoCitySchedCost(costs1))  # Output: 110

costs2 = [[259,770],[448,54],[926,667],[184,139],[840,118],[577,469]]
print(twoCitySchedCost(costs2))  # Output: 1859

costs3 = [[515,563],[451,713],[537,709],[343,819],[855,779],[457,60],[650,359],[631,42]]
print(twoCitySchedCost(costs3))  # Output: 3086



# In[8]:


##merge sorted array
def merge(nums1, m, nums2, n):
    # Start merging from the end of nums1
    while m > 0 and n > 0:
        if nums1[m - 1] > nums2[n - 1]:
            nums1[m + n - 1] = nums1[m - 1]
            m -= 1
        else:
            nums1[m + n - 1] = nums2[n - 1]
            n -= 1
    
    # If there are still elements remaining in nums2, copy them to nums1
    while n > 0:
        nums1[n - 1] = nums2[n - 1]
        n -= 1

# Example usage:
nums1 = [1,2,3,0,0,0]
m = 3
nums2 = [2,5,6]
n = 3
merge(nums1, m, nums2, n)
print(nums1)  # Output: [1,2,2,3,5,6]

nums1 = [1]
m = 1
nums2 = []
n = 0
merge(nums1, m, nums2, n)
print(nums1)  # Output: [1]

nums1 = [0]
m = 0
nums2 = [1]
n = 1
merge(nums1, m, nums2, n)
print(nums1)  # Output: [1]



# In[9]:


##insertion sort list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insertionSortList(head):
    if not head or not head.next:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    prev_sorted = head
    curr = head.next
    
    while curr:
        if prev_sorted.val <= curr.val:
            prev_sorted = prev_sorted.next
        else:
            prev = dummy
            while prev.next.val <= curr.val:
                prev = prev.next
            prev_sorted.next = curr.next
            curr.next = prev.next
            prev.next = curr
        
        curr = prev_sorted.next
    
    return dummy.next

# Utility function to convert list to linked list
def list_to_linked_list(lst):
    if not lst:
        return None
    head = ListNode(lst[0])
    current = head
    for val in lst[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# Utility function to convert linked list to list
def linked_list_to_list(head):
    lst = []
    current = head
    while current:
        lst.append(current.val)
        current = current.next
    return lst

# Example usage:
head1 = list_to_linked_list([4,2,1,3])
sorted_head1 = insertionSortList(head1)
print(linked_list_to_list(sorted_head1))  # Output: [1,2,3,4]

head2 = list_to_linked_list([-1,5,3,4,0])
sorted_head2 = insertionSortList(head2)
print(linked_list_to_list(sorted_head2))  # Output: [-1,0,3,4,5]


# In[10]:


##kth smallest element in a sorted matrix
def count_less_equal(matrix, target):
    count = 0
    n = len(matrix)
    row = n - 1
    col = 0
    
    while row >= 0 and col < n:
        if matrix[row][col] <= target:
            count += row + 1
            col += 1
        else:
            row -= 1
    
    return count

def kthSmallest(matrix, k):
    n = len(matrix)
    start = matrix[0][0]
    end = matrix[n - 1][n - 1]
    
    while start < end:
        mid = start + (end - start) // 2
        count = count_less_equal(matrix, mid)
        if count < k:
            start = mid + 1
        else:
            end = mid
    
    return start

# Example usage:
matrix1 = [[1,5,9],[10,11,13],[12,13,15]]
k1 = 8
print(kthSmallest(matrix1, k1))  # Output: 13

matrix2 = [[-5]]
k2 = 1
print(kthSmallest(matrix2, k2))  # Output: -5


# In[ ]:




