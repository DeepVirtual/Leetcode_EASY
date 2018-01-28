# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:38:57 2017

@author: cz
"""

#1. Two Sum

class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(nums)<=1:
            return None
        
        dictIndex={}
        
        for i,elem in enumerate(nums):
            
            m=target-elem
            
            if m in dictIndex:
                return [dictIndex[m],i]
            else:
                dictIndex[elem]=i
        return None

s=Solution()

t=s.twoSum([2, 7, 11, 15],9)

#7. Reverse Integer

class Solution:
     
   def reverse(self, x):
       """
       :type x: int
       :rtype: int
       """
       if x<0:
          return -self.reverse(-x)      
       ans=0      
       while x:
          
             ans=ans*10+x%10
             x=x//10
       return ans if ans < 0x7FFFFFFF else 0
  

s=Solution()

t=s.reverse(3456)         
        
          
#9. Palindrome Number      
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x<0:
            return False
        div=1
        while x//div >=10:
            div=div*10
            
        while x:
            left=x//div
            right=x%10
            
            if left !=right:
                return False
        
            x= (x%div)//10
            div=div/100
        return True
        
s=Solution()

t=s.isPalindrome(343)         
                
#13. Roman to Integer       
#http://blog.sina.com.cn/s/blog_7025794a0101397g.html

class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        numerals = { "M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1 }
        
        s=s[::-1]
        
        last=None
        sumRM=0
        for x in s:
            if last and numerals[x]<last:
                sumRM-=numerals[x]*2
            
            sumRM+=numerals[x]
            last=numerals[x]
        return sumRM
    

s=Solution()

t=s.romanToInt('IV')          
        
        
        
 
#14. Longest Common Prefix     
        
#        
#        
#strs=['qw456','wert','sdf']       
#for i,x in enumerate(zip(*strs)):
#  print(i,x)      
        
        
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs)==0:
            return ""
        if len(strs)==1:
            return strs[0]
        
        for i in range(len(strs[0])):
            for string in strs[1:]:
                if i>=len(string) or strs[0][i]!=string[i]:
                   return strs[0][:i]
        return strs[0]
        

if __name__ == "__main__":
    print(Solution().longestCommonPrefix(["abab","aba",""]))       


#20. Valid Parentheses
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s)==0:
            return False
        stack=[]
        
        dicts={ ')': '(',  '}': '{',  ']' :'['}
        for char  in s:
            if char in dicts.values():
                stack.append(char)
            elif char in dicts.keys():
                if not stack or dicts[char] != stack.pop():
                    return False
            else:
                return False
            
        return not stack
                
                
if __name__ == "__main__":
    print(Solution().isValid('[]{{'))                       
    
#21. Merge Two Sorted Lists 
    
# Definition for singly-linked list.
class ListNode:
     def __init__(self, x):
         self.val = x
         self.next = None
     def __repr__(self):
        if self:
            return "{} -> {}".format(self.val, self.next)
        
class Solution:
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
        if not l1:
            return l2
        if not l2:
            return l1
        
        dummy=cur=ListNode(0)
        
        while  l1 and l2:
            print(cur)
            if l1.val>=l2.val:
                cur.next=l2
                l2=l2.next
            else:
                cur.next=l1
                l1=l1.next
            cur=cur.next
            
        if  l1:
            cur.next=l1
        if  l2:
            cur.next=l2
        return dummy.next
    
    
if __name__ == "__main__":
    l1 = ListNode(1)
    l1.next = ListNode(9)
    print(l1)
    l2 = ListNode (2)
    l2.next = ListNode(3)
    print(l2)
    print(Solution().mergeTwoLists(l1, l2))    
    
    
#26. Remove Duplicates from Sorted Array    
    
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if not nums:
            return 0
        end=0
        
        for i in range(len(nums)):
            if nums[i]!=nums[end]:
               nums[end+1]= nums[i]
               end+=1
                
        print(nums[:end+1])   
        return end+1
    
if __name__ == "__main__":
    print(Solution().removeDuplicates([1,2,3,3,3,4,3,4,4,0,4,4]))                    
    
    
#27. Remove Element    

class Solution:
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        
        if not nums:
            return 0
        
             
        i=0
        n=len(nums)
        
        while i<n:
            if nums[i]==val:
                nums[i]=nums[n-1]
                n-=1
            else:
                i+=1
        print(i,n)
        print(nums[:i])
        return i
                
if __name__ == "__main__":
    print(Solution().removeElement([1,2,3,3,3,4,3,4,4,0,4,4],3))               
    
    
    
#28. Implement strStr()   
class Solution:
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        
        h=len(haystack)
        n=len(needle)
        
        
        for i in range(h-n+1):
            if haystack[i:i+n]==needle:
                return i
        return -1
if __name__ == "__main__":
    print(Solution().strStr("aaaaa","bba"))              
            
            
#35. Search Insert Position            
        
class Solution:
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """ 
        if not nums:
            return 0
        
        left=0
        right=len(nums)-1
        
        
        while left <= right:
            
            mid=(left+right)//2
            
            if nums[mid]<target:
                left=mid+1
            elif nums[mid]>target:
                right=mid-1
            else:
                return mid
        return left
    
if __name__ == "__main__":
    print(Solution().searchInsert([1,3,5,6], 7))      
    
    
#38. Count and Say    
class Solution:
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """         
        if n==0:
            return ''        
        s='1'        
        for i in range(1,n):
            s=self.cal(s)
        return s  
    
    def cal(self,s):
        length=len(s)
        count=1
        ans=''
        
        for i,c in enumerate(s):
            if i+1 <length and s[i]!=s[i+1]:
                ans=ans+str(count)+c
                count=1
            elif i+1 <length:
                 count+=1
                 
        return ans+str(count)+c
            
if __name__ == "__main__":
    print(Solution().countAndSay(3))         
        
        
#53. Maximum Subarray           
    
class Solution:
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
            
        if not nums:
           return None
       
        sumS=0
       
        maxS=-float("inf")
        
        for i in range(len(nums)):
            sumS=sumS+nums[i]
            maxS=max(maxS,sumS)
            if sumS<0:
                sumS=0
        return maxS
                
if __name__ == "__main__":
    print(Solution().maxSubArray([1,3,4,1,-10,1,5,4]))         
                    
#58. Length of Last Word    
class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        listS=s.strip().split()
        
              
        return len(listS[-1]) if listS else 0
    
if __name__ == "__main__":
     print(Solution().lengthOfLastWord("Hello World"))      
    
    
    
#66. Plus One    
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        flag=1
        
        for i in range(len(digits)-1,-1,-1):
            if digits[i]+flag==10:
                digits[i]=0
                flag=1
                
            else:
                digits[i]+=flag
                flag=0
                
        if flag==1:
            digits.insert(0,1)
            
        return digits
 
        
#67. Add Binary       
class Solution:
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if len(a)==0:
            return b
        if len(b)==0:
            return a
        
        if a[-1]=='1' and b[-1]=='1':
            return self.addBinary(self.addBinary(a[:-1],b[:-1]),'1')+'0'
        elif a[-1]=='0' and b[-1]=='0':
            return self.addBinary(a[:-1],b[:-1])+'0'
        else:
            return self.addBinary(a[:-1],b[:-1])+'1'
        
        
if __name__ == "__main__":
     print(Solution().addBinary('110','101'))                  
            
#69. Sqrt(x)            
class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        r=x
        while r*r>x:
            r=(r+x//r)//2
        return r
if __name__ == "__main__":
     print(Solution().mySqrt(10))                  
                              
            
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """

        dp=[0]*(n+1)
        dp[0]=0
        if n>=1:
           dp[1]=1
        if n>=2:
           dp[2]=2
        if n>=3:
           for i in range(3,n+1):
              dp[i]=dp[i-1]+dp[i-2]
        return dp[n]
if __name__ == "__main__":
     print(Solution().climbStairs(3))   
     
     
#83. Remove Duplicates from Sorted List     
       
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """       
        
        curr=head
        
        while curr and curr.next:
            
            if curr.val==curr.next.val:
                curr.next=curr.necx.next
                
            else:
                curr=curr.next
        return head
                
            
#88. Merge Sorted Array          
            
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        
        
        while n>0 and m>0:
            if nums1[m-1]>=nums2[n-1]:
                nums1[m+n-1]= nums1[m-1]
                m=m-1
            else:
                nums1[m+n-1]= nums2[n-1]
                n=n-1
        if n>0:
            
            nums1[:n]=nums2[:n]           
            
            
# 100. Same Tree            
            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p and q:
            return p.val==q.val and  self.isSameTree(p.left,q.left)  and  self.isSameTree(p.right,q.right)
        return p is q         
            
            
            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        if not root:
            return True
        
        return  self.isMirror(root,root) 
    
    def isMirror(self,node1,node2):
        if(node1 is None  and node2 is None):
            return True
        if(node1 is None  or node2 is None): 
            return False
        
        return node1.val==node2.val and self.isMirror(node1.left,node2.right) and self.isMirror(node1.right,node2.left)
            
            
#104. Maximum Depth of Binary Tree            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
#        if not root.left  and not root.right:
#            return 0
        
        depth=1+max ( self.maxDepth(root.left) ,   self.maxDepth(root.right))
        
        return depth
            
            
#107. Binary Tree Level Order Traversal II            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res=[]
        stack=[(root,0)]
        
        while stack:
            node,level=stack.pop()
            if node:
               if len(res)<level+1:
                      res.insert(0,[])
            
               res[-(level+1)].append(node.val)
          
               stack.append((node.left,level+1))
               stack.append((node.right,level+1))
        return res


#108. Convert Sorted Array to Binary Search Tree

# Definition for a binary tree node.
class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Solution:
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
           return None
    
        mid=(len(nums)-1)//2
    
        root=TreeNode(nums[mid])
        root.left=self.sortedArrayToBST(nums[:mid])
        root.right=self.sortedArrayToBST(nums[mid+1:])
        
        return root
    
    
#110. Balanced Binary Tree    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool

        """    
        if not root:
           return True
       
        diff= abs(self.Depth(root.left)-self.Depth(root.right))
        return diff <=1 and self.isBalanced(root.left) and self.isBalanced(root.right)
    
    def Depth(self,Node):
        if not Node:
            return 0
        return 1+max(self.Depth(Node.left),self.Depth(Node.right))
        
       
#111. Minimum Depth of Binary Tree        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        right=self.minDepth(root.right)
        left=self.minDepth(root.left)
        
        if right and left:            
            return min(right,left)+1        
        else:
            return right+left+1
        
        
        
#112. Path Sum   
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        
        if not root:
            return False
        
        if not root.left  and not root.right and root.val==sum:
            return True
        
        sum-=root.val
        
        return self.hasPathSum(root.left,sum)  or self.hasPathSum(root.right,sum)
        
        
#118. Pascal's Triangle        
    
class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        res=[[1]]
            
        for i in range(numRows):
            temp1=res[-1]+[0]
            temp2=[0]+res[-1]
            temp3=[list(map(lambda x, y: x+y,temp1,temp2))]
            res+=temp3
        
        return res[:numRows]
                
if __name__ == "__main__":
     print(Solution().generate(0))     
    
    
#119. Pascal's Triangle II    
class Solution:
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        
        
        row=[1]
        
        for _ in range(rowIndex):
            row=[x+y for x,y in zip([0]+row,row+[0])]
        return row
    
if __name__ == "__main__":
     print(Solution().getRow(3))        
        
#121. Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        
        maxProfit=0
        minPrice=float("inf")
        
        
        for i in range(len(prices)):
            if prices[i]<minPrice:
               minPrice=prices[i]
            elif prices[i]-minPrice>maxProfit:
                maxProfit=prices[i]-minPrice
        return maxProfit
                
                 
#122. Best Time to Buy and Sell Stock II     
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        if not prices:
            return 0
        
        
        maxProfit=0
        for i in range(1,len(prices)):
            if prices[i]>prices[i-1]:
                print(prices[i],prices[i-1])
                maxProfit+=prices[i]-prices[i-1]     
        
        
            
        return  maxProfit
        
        
if __name__ == "__main__":
     print(Solution().maxProfit([7, 1, 5, 3, 6, 4]))                
     
        
        
#125. Valid Palindrome        
class Solution:
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        l=0
        r=len(s)-1
        
        while l<r:
            
            while l<r and not s[l].isalnum():
                l+=1
            
            while l<r and not s[r].isalnum():
                r-=1
            if s[l].lower() !=s[r].lower() :
                return False
            
            l+=1
            r-=1
            
        return True

if __name__ == "__main__":
     print(Solution().isPalindrome("race a car")) 

        
        
#136. Single Number        
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        hash_table={} 
        
        for i in nums:
            try:
                hash_table.pop(i)
            except KeyError:
                hash_table[i]=1
        return hash_table.popitem()[0]
    
    
        
#141. Linked List Cycle        
        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """        
        if not head or not head.next:
            return False
        
        fast=slow=head
        
        
        while fast.next and  fast.next.next:
              fast=fast.next.next
              slow=slow.next
              if fast is slow:
                  return True
              
              
        return False
        
#155. Min Stack        
# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()          
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.q=[]
        

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        curmin=self.getMin()
        if  curmin is None or curmin>x:
            curmin=x
        self.q.append((x,curmin))
        

    def pop(self):
        """
        :rtype: void
        """
        self.q.pop()

    def top(self):
        """
        :rtype: int
        """
        
        if not self.q:
            return None
        return self.q[-1][0]
        

    def getMin(self):
        """
        :rtype: int
        """
        if not self.q:
            return None
        return self.q[-1][1]

#160. Intersection of Two Linked Lists
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        
        if not headA or not headB:
               return None
           
        runA=headA
        runB=headB
        
        while runA!=runB:
            if not runA:
               runA=headB
            elif not runB:
                 runB=headA
            else:
                 runA=runA.next
                 runB=runB.next
        return runA
            
     
#167. Two Sum II - Input array is sorted     
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """     
        d={}
        
        for i , item in enumerate(numbers):
            if target-item in d:
                return (d[target-item]+1,i+1)
            else:
                d[item]=i
     
#168. Excel Sheet Column Title     
class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        res='' 
        while n >0: 
             res=chr((n - 1) % 26 + ord('A'))+res
             #print(n)
             n=(n-1)//26
             
        return res
    
if __name__ == "__main__":
     print(Solution().convertToTitle(27)) 
     
     
#169. Majority Element 
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
               
        d={}
        
        for num in nums:
            if num not in d:
                d[num]=1
            if d[num] >len(nums)/2:
                return num
            else:
                d[num]+=1
            
            
#171. Excel Sheet Column Number            
     
class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        n=0
        
        for i  in range(len(s)):
            
             #print(s[i],n)
             
             n=n*26+ord(s[i])-65+1
             #print(s[i],n)
        return n
            
if __name__ == "__main__":
     print(Solution().titleToNumber('Z'))             
     
     
#172. Factorial Trailing Zeroes     
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0:
            return 0
        else:
            return n//5+self.trailingZeroes( n//5)
        
        
#175. Combine Two Tables        
#select FirstName, LastName, City, State  from Person  
#left join Address
#on Person.PersonId=Address.PersonId  

#176. Second Highest Salary

#select max(Salary) as  SecondHighestSalary from Employee
#where Salary < ( select max(Salary)  from Employee)

#181. Employees Earning More Than Their Managers
#
#select a.Name  as  Employee from Employee a
#join Employee b
#on a.ManagerId=b.Id
#where a.Salary>b.salary
#      
            
        
#182. Duplicate Emails        
#select  Email  from Person
#group by Email
#having count(*)>1        
        
#183. Customers Who Never Order   
# Write your MySQL query statement below
#select Name  as Customers  from Customers
#where Id not in
#( select CustomerId from Orders) 
            
        
#189. Rotate Array        
        
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if not nums:
            return None
        k=k%len(nums)
        n=len(nums)
        self.reverse(nums,0,n-1)
        self.reverse(nums,0,k-1)
        self.reverse(nums,k,n-1)
       # print(nums)
        
    def reverse(self,lst,start,end):
        while start<end:
            lst[start],lst[end]=lst[end],lst[start]
            start+=1
            end-=1
            
        
        
        
if __name__ == "__main__":
     print(Solution().rotate([1,2,3,4,5,6,7],3))
                  
#190. Reverse Bits        
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        
        return  int(bin(n)[2:].zfill(32)[::-1],2)


        
#191. Number of 1 Bits        
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count=0
        mask=1
        for _ in range(32):
            
            if (n & mask) !=0:
                #print(n,mask,n & mask)
                count+=1
            mask=mask << 1
        return count
    
if __name__ == "__main__":
     print(Solution().hammingWeight(3))                
                
#193. Valid Phone Numbers      

#sed -r -n '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/p'  file.txt  

#195. Tenth Line
#sed -n 10p file.txt

#196. Delete Duplicate Emails

#delete p1 from
#Person p1,
#Person p2
#
#where p1.Email =p2.Email
#and p1.Id > p2.Id

#197. Rising Temperature
#select w2.Id from Weather w1, Weather w2
#where  w2.Temperature >w1.Temperature
#and  TO_DAYS (w2.Date) -  TO_DAYS (w1.Date)=1     



#198. House Robber
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        last=0
        cur=0
        
        for num in nums:
            last, cur=cur, max(cur,last+num)
        
        return cur

#202. Happy Number
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        
        fast=self.calSum(n)
        slow=n      
        if n==1:
            return True
        while fast != slow:
              #print(fast,slow)
              fast= self.calSum( self.calSum(fast))
              slow=self.calSum(slow)
              #print(fast,slow)
              if slow==1 or fast==1:
                  return True            
            
        return False
        
        
    def calSum(self,m):
        sumD=0
            
        while m>0:
              sumD+=(m%10)*(m%10)
            
              m=m//10
        return sumD
    
    
    
    
        
if __name__ == "__main__":
     print(Solution().isHappy(2))          
        
        
#203. Remove Linked List Elements        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy=ListNode(-1)
        dummy.next=head
        cur=dummy
        
        while cur and cur.next:
            if cur.next.val==val:
                cur.next=cur.next.next
            else:
                cur=cur.next
        
        return dummy.next
        
#204. Count Primes    
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        if n <= 2:
            return 0
        
        

        listN=[0]*n
        
        count=0
        

        for i in range(2,n):
           j=2
           if not listN[i]:
               count+=1
               while i *j <n:
                    
                    listN[i*j]=True
                    j+=1
                   
          
        return count
        
if __name__ == "__main__":
     print(Solution().countPrimes(10))          
                
        
#205. Isomorphic Strings        
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """        
        d1,d2={},{}
        
        for ss , tt in zip(s,t):
            if ss not in d1.keys():
                d1[ss]=tt
            if tt not in d2.keys():
                d2[tt]=ss
            if d1[ss]!=tt or d2[tt]!=ss:
                return False
        return True
            
if __name__ == "__main__":
     print(Solution().isIsomorphic("paper", "title"))          
                         
#206. Reverse Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        prev=None
        cur=head
        
        while cur and cur.next:
              tmp= cur.next
              cur.next=prev
              prev=cur
              cur=tmp
        return cur

#217. Contains Duplicate
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        hashSet=set()
        
        for num in nums:
            if num not in hashSet:
                hashSet.add(num)
            else:
                return True
        return False
        
            
#219. Contains Duplicate II            
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
                    
        hashSet={} 
        
        for  i,item in enumerate(nums):
             if item in hashSet.keys() and abs(i-hashSet[item])<=k:
                 return True
             else:
                 hashSet[item]=i
                 
        return False
        
#225. Implement Stack using Queues

from collections import deque 
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._queue=deque()
        self.size=0
        

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self._queue.append(x)
        self.size+=1       
        

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        self.size-=1
        return self._queue.pop()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self._queue[-1]
    
    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return self.size==0


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()        
        
#226. Invert Binary Tree        
 # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        
        if not root:
            return None
        
        right=self.invertTree(root.right)
        left=self.invertTree(root.left)
        
        root.left=right
        root.right=left
        
        return root
        
#231. Power of Two        
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """

        if n<1:
            return False
        
        if n==1:
            return True
        
        
        return self.isPowerOfTwo(n/2)
            
if __name__ == "__main__":
     print(Solution().isPowerOfTwo(3))                    
            
     
#232. Implement Queue using Stacks     
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.inStack,self.outStack=[],[]
        self.size=0
        

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.size+=1
        self.inStack.append(x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        self.move()
        self.size-=1
        return self.outStack.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        
        self.move()
        return self.outStack[-1]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return self.size==0
    
    def move(self):
        if not self.outStack:
            while self.inStack:
                self.outStack.append(self.inStack.pop())
            

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()     
     
#234. Palindrome Linked List     
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """   
        if not head:
            return True
        
        fast=slow=head
        
        while fast and fast.next:
            slow=slow.next
            fast=fast.next.next
            
        #reverse the second half of the linked list
        
        prev=None
        cur=slow
        while cur:
              tmp=cur.next
              cur.next=prev
              prev=cur
              cur=tmp
        # prev is the new head
        p1=prev
        p2=head
        
        #check if isPalindrome
        while p1 and p2 :
            
            if p1.val!= p2.val:
                return False
            p1=p1.next
            p2=p2.next
        return True
            
            
#235. Lowest Common Ancestor of a Binary Search Tree            
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        while (root.val-  p.val) * (root.val-  q.val)>0:
            
            if root.val>p.val:
              root=root.right
            else:
                root=root.left
        return root
        
#237. Delete Node in a Linked List        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        tmp=node.next
        node.next=tmp.next
        node.val=tmp.val
        
#242. Valid Anagram        
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """        
        if len(s)==0 and len(t)==0:
            return True
        
        if len(s) != len(t):
            
            return False 
        
        hashDict={}
        
        for i in range(len(s)):
            if s[i] not in hashDict:
                hashDict[s[i]]=1
            else:
                hashDict[s[i]]+=1
        print()
        for j in range(len(t)):
            if t[j]  not in hashDict:
                return False
            else:
                hashDict[t[j]]-=1
        
        for key,item in hashDict.items():
            if item!=0:
                return False
        return True
if __name__ == "__main__":
     print(Solution().isAnagram("aacc","ccac"))        
        
        
#257. Binary Tree Paths        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        #DFS  Stack
        
        if not root:
            return []
        
        stack=[]
        ls=''
        res=[]
        stack.append((root,ls))
        
        while stack:
            node,ls=stack.pop()
            
            if not node.left and not node.right:
                res.append(ls+str(node.val))
            if node.left:
               stack.append((node.left,ls+str(node.val)+'->'))
               
            if node.right:
               stack.append((node.right,ls+str(node.val)+'->'))
               
        return res
               
        
#258. Add Digits        
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        
        if num==0:
            return 0
        if num%9==0:
            return 9
        if num%9!=0:
            return num%9
            
        
        
#263. Ugly Number        
class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num<=0:
            return False
        if num==1:
            return True
        
        while num!=1:
        
           if  num%2==0:
                 num=num/2
           elif  num%3==0:
                 num=num/3
           elif  num%5==0:
                 num=num/5
           else:
               return False
        return True
if __name__ == "__main__":
     print(Solution().isUgly(9))             
            
        
#268. Missing Number        
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        hashSet=set(nums)  
        
        for i in range(len(nums)):
            if i not in hashSet:
                return i
            
        return 0        
        
        
#278. First Bad Version        
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):
#bool isBadVersion(version)
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0 or n==1:
            return n
        
        return self.checkVersion(1,n)
        
    def checkVersion(self,l,r): 
        if l==r:
             return l
        
        mid=(l+r)//2
        if  isBadVersion(mid):
            self.checkVersion(l,mid)
        else:
            self.checkVersion(mid+1,r)
        
        
#283. Move Zeroes        
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        
        if len(nums)==0:
            return []
        
        count=0
        for i in range(len(nums)):
            if nums[i]!=0:
               nums[count]=nums[i]
               count+=1
                
        for j in range(count,len(nums)):
            nums[j]=0
        return nums
if __name__ == "__main__":
    print(Solution().moveZeroes([1]))            
            
            
#290. Word Pattern            
class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        strList=str.strip().split()
        
        if len(pattern)!=len(strList):
            return False
        
        d1={}
        d2={}
        
        for i in range(len(pattern)):
            if pattern[i] not in d1:
                d1[pattern[i]]=strList[i]
            elif d1[pattern[i]]!=strList[i]:
                return False
            
            if strList[i] not in d2:
                d2[strList[i]]=pattern[i]
            elif d2[strList[i]]!=pattern[i]:
                return False
                
        return True
if __name__ == "__main__":
    print(Solution().wordPattern("abba","dog dog dog dog"))            
                    
        
#292. Nim Game        
class Solution(object):
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n%4==0:
           return  False
        else:
            return True
        
#303. Range Sum Query - Immutable        
     


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)        
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.dp=nums
        for i in range(1,len(nums)):
            self.dp[i]=self.dp[i-1]+nums[i]
        
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        if i==0:
            return self.dp[j]-0
        else:
            return self.dp[j]-self.dp[i-1]
        

                                

#326. Power of Three        
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """       
        if n==1:
            return True
        elif n<1:
            return False
            
        return self.isPowerOfThree(n/3.0)
        
if __name__ == "__main__":
    print(Solution().isPowerOfThree(14348907))                    
        
        
#342. Power of Four        
class Solution(object):
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """        
        if num==1:
            return True
        elif num<1:
            return False
            
        return self.isPowerOfFour(num/4.0)
        
        
#344. Reverse String        
class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]
        
#345. Reverse Vowels of a String
class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """       
        if not s:
            return ''
        if len(s)==1:
            return s
        
        Vowels=['a','e','i','o','u','A','E','I','O','U']
        sL=list(s)
        l=0
        r=len(s)-1
        while l<r:
            
            while l<r and s[l] not in Vowels:
                l+=1
            while l<r and s[r] not in Vowels:
                r-=1    
    
            sL[l], sL[r]=sL[r], sL[l]
            
            l+=1
            r-=1
            
        return "".join(sL)
if __name__ == "__main__":
    print(Solution().reverseVowels("leotcede"))                    
            
#349. Intersection of Two Arrays
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        s1=set(nums1)
        s2=set(nums2)
        s=s1.intersection(s2)
        return list(s)
        
        
        
#350. Intersection of Two Arrays II       
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        
        d1={}
        d=[]
        
        if not nums1 or not nums2:
            return d
            
        
        
        for n in nums1:
            if n not in d1:
                d1[n]=1
            else:
                d1[n]+=1
            
        for m in nums2:
            if m in d1 and d1[m]>=1 :
                d.append(m)
                d1[m]-=1
        return d

if __name__ == "__main__":
    print(Solution().intersect([1, 2, 2, 1],[2, 2]))



#367. Valid Perfect Square
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        i=0
        while i*i<=num:
           if i*i==num:
               return True
           i+=1
        return False

if __name__ == "__main__":
    print(Solution().isPerfectSquare(7))


#371. Sum of Two Integers
class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
     # use bit operation
     # 32 bits integer max
        MAX = 0x7FFFFFFF
        mask = 0xFFFFFFFF
        #0xffffffff	is 11111111 11111111 11111111 11111111
        while b!=0:
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask
        return a if a< MAX else ~(a ^ mask)

   



if __name__ == "__main__":
    print(Solution().getSum(7,8))



#383. Ransom Note

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        from collections import Counter
        r=Counter(ransomNote)
        m=Counter(magazine)
        
        return not r-m

#374. Guess Number Higher or Lower
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        l=0
        r=n
        
        while l<r:
            
            mid=(l+r)//2
            
            if guess(mid)==0:
                return mid
            elif guess(mid)==-1:
                r=mid-1
            else:
                l=mid+1
        return l
        
#387. First Unique Character in a String        
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return -1
        letters='abcdefghijklmnopqrstuvwxyz'
        indexS=  [s.index(i) for i in letters if s.count(i)==1]   
        
        return min(indexS) if len(indexS)>0 else -1
        
        
        
#389. Find the Difference        
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        from collections import Counter
        
        sc=Counter(s)
        tc=Counter(t)
        
        return list(tc-sc)[0]
        
        
#400. Nth Digit       
class Solution(object):
    def findNthDigit(self, n):
        """
        :type n: int
        :rtype: int
        """
        size,step,start=1,9,1

        while n-size*step>0:
           n,size,step,start=n-size*step,size+1,step*10,start*10
           
        number=start+(n-1)//size
        digit=int(str(number)[(number-1)%size])
        return digit
           
#401. Binary Watch        
class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        return ['%d:%02d' % (h,m)  for h in range(12) for m in range(60) if (bin(h)+bin(m)).count('1')==num]
        
   
#404. Sum of Left Leaves        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if root.left and not root.left.left and not root.left.right:
            return root.left.val+self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.right)+self.sumOfLeftLeaves(root.left)

#405. Convert a Number to Hexadecimal
        #Two's Complement binary for Negative Integers:

#Negative numbers are written with a leading one instead of a leading zero. 
#So if you are using only 8 bits for your twos-complement numbers, then you treat patterns
# from "00000000" to "01111111" as the whole numbers from 0 to 127, and reserve "1xxxxxxx" 
# for writing negative numbers. A negative number, -x, is written using the bit pattern for (x-1) 
# with all of the bits complemented (switched from 1 to 0 or 0 to 1). So -1 is 
# complement(1 - 1) = complement(0) = "11111111", 
# and -10 is complement(10 - 1) = complement(9) = complement("00001001") = "11110110". 
# This means that negative numbers go all the way down to -128 ("10000000").
#Of course, Python doesn't use 8-bit numbers. 
#It USED to use however many bits were native to your machine,
# but since that was non-portable, it has recently switched to using an INFINITE number of bits. 
# Thus the number -5 is treated by bitwise operators as if it were written "...1111111111111111111011".
 
class Solution(object):
    def toHex(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num==0:
            return '0'
        
        res=''
        mp=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
        
        for _ in range(8):
            print(num)
            res=mp[num&15]+res
            num=num >> 4
        return res.lstrip('0')
    
#-1>>4 is always -1
    
if __name__ == "__main__":
    print(Solution().toHex(-1))


#409. Longest Palindrome
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import Counter
        
        if not s:
            return 0
        if len(s)==1:
            return 1
        
        t=Counter(s)
        
        res=0
        counto=0
        for k,v in t.items():
            if v%2==0:
                res+=v
            if v%2==1 :
                res+=v-1
                counto+=1
                
        return res+1 if counto>0 else res
if __name__ == "__main__":
    print(Solution().longestPalindrome("abccccdd"))        
        
        
        
        
#412. Fizz Buzz      
class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        if n==0:
            return []
        res=[]
        for i in range(1,n+1): 
            if i %15==0:
                res.append("FizzBuzz")
            elif i %5==0:
                res.append("Buzz")
        
            elif i %3==0:
                res.append("Fizz")
            else:
                res.append(str(i))
        return res
if __name__ == "__main__":
    print(Solution().fizzBuzz(15))        
                
        
        
#414. Third Maximum Number       
class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if not nums:
            return None
        
        res=[-float('inf'),-float('inf'),-float('inf')]
        for num in nums:
            if num not in res:
                if num >  res[0]:
                   res[0],res[1], res[2]=num,res[0],res[1]
                elif num>res[1]:
                   res[1],res[2]=num,res[1]
                elif num>res[2]:
                   res[2]=num
        print(res)
        return max(res) if -float('inf') in res else res[2]
if __name__ == "__main__":
    print(Solution().thirdMax([2, 2, 3, 1]))                
                
#415. Add Strings                
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
                
        if not num1:
            return num2
        
        if not num2:
            return num1

        n1=len(num1)-1
        n2=len(num2)-1
        carry=0        
        res=[]        
        while  n1>=0 or n2>=0 or carry>0:
               print(n1,n2)               
               if n1>=0:
                  n11=int(num1[n1])
               else:
                  n11=0               
               if n2>=0:
                   n22=int(num2[n2])
               else:
                  n22=0 
               res.append(str((n11+n22+carry)%10))
               carry=(n11+n22+carry)//10
               n1-=1
               n2-=1               
        return ("".join(res))[::-1]
                
if __name__ == "__main__":
    print(Solution().addStrings(['2', '2', '3', '1'],['9','9','9','9']))                   
                
                
                
#434. Number of Segments in a String                
class Solution(object):
    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """      
        if not s:
            return 0
        return len(s.slpit())
                
#437. Path Sum III        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """   
        
        if root:
            return self.findPath(root,sum)+self.pathSum(root.left,sum)+self.pathSum(root.right,sum)
        return 0
    
    def findPath(self,root,sum):
        
        
        if root:
            return int(root.val==sum)+self.findPath(root.left,sum-root.val)+self.findPath(root.right,sum-root.val)
        return 0
        

        
#438. Find All Anagrams in a String        
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        from collections import Counter
        n=len(p)
        
        sCounter=Counter(s[:n-1])
        pCounter=Counter(p)
        
        res=[]
        
        for i in range(n-1,len(s)):
            sCounter[s[i]]+=1
            if sCounter==pCounter:
                res.append(i-n+1)
            
            sCounter[s[i-n+1]]-=1
            
            if sCounter[s[i-n+1]]==0:
                del sCounter[s[i-n+1]]
        return res
        
        
#441. Arranging Coins        
class Solution(object):
    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0:
            return 0
        total=0
        i=1
        
        while total<n:
              print(i,total)
              total+=i
              i+=1
              
        
        if total==n:
            return i-1
        if total>n:
            return i-2
            
if __name__ == "__main__":
    print(Solution().arrangeCoins(5))                   
                            
            
#443. String Compression            
class Solution(object):
    def compress(self, chars):
        """
        :type chars: List[str]
        :rtype: int
        """
        
        write=anchor=0
        for read,c in enumerate(chars):
            if read+1==len(chars)  or chars[read+1]!=c:
                chars[write]=chars[anchor]
                write+=1
                if read>anchor:
                    for digit in str(read-anchor+1):
                        chars[write]=digit
                        write+=1
                anchor=read+1
        return write
            
        
#447. Number of Boomerangs        
class Solution(object):
    def numberOfBoomerangs(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        
        
        res=0
        for p in points:
            distance={}
            for q in points:
                f=p[0]-q[0]
                s=p[1]-q[1]
                distance[f*f+s*s]=1+distance.get(f*f+s*s,0)
                
            for k in distance:
                res+=distance[k]*(distance[k]-1)
        return res
                
        
        
if __name__ == "__main__":
    print(Solution().numberOfBoomerangs([[0,0],[1,0],[2,0]]))        
        
        
#448. Find All Numbers Disappeared in an Array      
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        if not nums:
            return []
        
        res=[]
        for i in range(len(nums)):
            index=abs(nums[i])-1
            nums[index]=-abs(nums[index])
            print(i,index,nums)
        for i in range(len(nums)):
            
            if nums[i]>0:
                res.append(i+1)
        return res
            
if __name__ == "__main__":
    print(Solution().findDisappearedNumbers([4,3,2,7,8,2,3,1]))            
        
#453. Minimum Moves to Equal Array Elements        
class Solution(object):
    def minMoves(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """        
#A move can be interpreted as: "Add 1 to every element and subtract one from any one element". sum(nums_new) = sum(nums) + (n-1): we increment only (n-1) elements by 1.
#Visualize the nums array as a bar graph where the value at each index is a bar of height nums[i]. We are looking for minimum moves such that all bars reach the final same height.
#Now adding 1 to all the bars in the initial state does not change the initial state - it simply shifts the initial state uniformly by 1.This gives us the insight that a single move is equivalent to subtracting 1 from any one element with respect to the goal of reaching a final state with equal heights.
#So our new problem is to find the minimum number of moves to reach a final state where all nums are equal and in each move we subtract 1 from any element.
#The final state must be a state where every element is equal to the minimum element. Say we make K moves to reach the final state. Then we have the equation, N * min(nums) = sum(nums) - K.
##        
        return sum(nums)-min(nums)*len(nums)
        
        
#455. Assign Cookies        
class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """        
        g.sort()# child
        s.sort()#cookie
        
        ichild=0
        icookie=0
        
        
        while ichild<len(g)  and icookie<len(s):
             if s[ icookie]>=g[ichild]:
                   ichild+=1
             
             icookie+=1
        return ichild
            
#458. Poor Pigs            
class Solution(object):
    def poorPigs(self, buckets, minutesToDie, minutesToTest):
        """
        :type buckets: int
        :type minutesToDie: int
        :type minutesToTest: int
        :rtype: int
        """
        import math
        pigs=0

        while  (math.ceil(minutesToTest/minutesToDie+1))**pigs< buckets:
               pigs+=1
        return pigs
            
#459. Repeated Substring Pattern            
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """            
        return True if s in (s+s)[1:-1] else False
        
#461. Hamming Distance        
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        return bin(x^y).count('1')

#463. Island Perimeter
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        if not grid:
            return 0
        
        island=0
        neighbor=0
        for i in range(len(grid)):
            for j in range(len(grid[0])): 
                if grid[i][j]==1:
                   island+=1
                   if i+1<len(grid) and grid[i+1][j]==1:
                      neighbor+=1
                   if j+1<len(grid[0]) and grid[i][j+1]==1:
                      neighbor+=1
                #print(i,j,island,neighbor)
        return 4*island-2*neighbor
    
if __name__ == "__main__":
    print(Solution().islandPerimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]))            
                            
      
#475. Heaters        
class Solution(object):
    def findRadius(self, houses, heaters):
        """
        :type houses: List[int]
        :type heaters: List[int]
        :rtype: int
        """       
        import bisect
        
        ans=0
        heaters.sort()
        
        for house in houses:
            hi=bisect.bisect_left( heaters,house)
            left=heaters[hi-1] if hi-1>=0 else float('-inf')
            right=heaters[hi] if hi<len(heaters) else float('inf')
            ans=max(ans,min(house-left,right-house))
        return ans
        
        
#476. Number Complement        
class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """        
        mask=1
        
        mask=1<<(len(bin(num))-2)
        
        return (mask-1)^num
        
        
        
#479. Largest Palindrome Product        
class Solution(object):
    def largestPalindrome(self, n):
        """
        :type n: int
        :rtype: int
        """
#        if n==1:
#            return 9
#        upbound=10**n-1 
#        lowbound=upbound//10
#        maxNumber=upbound*upbound
#        firsthalf=maxNumber//(10**n)
#        palindromFound = False
#        while not palindromFound :
#            Palindrome=self.createPalindrome(firsthalf)
#            for i in range(upbound,lowbound,-1):
#                if Palindrome/i >upbound or i*i<Palindrome:
#                    break
#                if Palindrome%i==0:
#                    palindromFound=True
#                    break
#            firsthalf-=1
#        return Palindrome % 1337 
#        
#    def createPalindrome(self,m):
#        s=str(m)
#        Palindrome=int(s+s[::-1])
#        return Palindrome
        return [9, 9009, 906609, 99000099, 9966006699, 999000000999, \
                    99956644665999, 9999000000009999][n - 1] % 1337       
if __name__ == "__main__":
    print(Solution().largestPalindrome(7))        
        
        
        
#        maxN=0
#        for i in range(10**n-1,10**(n-1),-1):
#            
#            for j in range(i,10**(n-1),-1):
#                
#                m=i*j
#                #print(maxN)
#                if self.isPalindrome(m):
#                    maxN=max(maxN,m)
#                    #print(m)
#        return maxN%1337
            
            
            
#    def isPalindrome(self,m):
#        mlist=[]
#        
#        while m>0:            
#          mlist.append(m%10)
#          m=m//10
#        #print(mlist)
#        i=0
#        while i in range(len(mlist)//2):
#            if mlist[i]!=mlist[-(i+1)]:
#                return False
#            i+=1
#        return True
        
if __name__ == "__main__":
    print(Solution().largestPalindrome(2))            
    #print(Solution().isPalindrome(9009))                               
        
#482. License Key Formatting  
class Solution:
    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        
        
   
        ls=''.join(S.split('-'))
        if not ls:
            return ''
        
        n=len(ls)
        
        remainder=n%K
        
        if remainder>0:
           first=ls[:remainder]+'-'
        else:
           first=ls[:remainder]
           
        rest=[]
        for i in range(remainder,n):
            rest.append(ls[i])
            print(i,K,remainder)
            if (i-remainder+1)%K==0 and i!=n-1:
               
                rest.append('-')
        res=first.upper()+ ''.join(rest).upper() 
        
        return res if res[-1]!='-' else res[:-1]  
S=  "5F3Z-2e-9-w"
S=  "2-5g-3-J"
S='2'
S='---'
K=3 
if __name__ == "__main__":
    print(Solution().licenseKeyFormatting(S, K))
    
    

#485. Max Consecutive Ones        
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if not nums:
           return 0
        
        maxCount=0
        count=0
        for num in nums:
            if num==1:
               count+=1
            else:
               
               maxCount=max(count,maxCount)
               count=0
            
        return max(count,maxCount)
                           
if __name__ == "__main__":
    print(Solution().findMaxConsecutiveOnes([1,1,0,1,1,1]))            
            
#492. Construct the Rectangle        
class Solution(object):
    def constructRectangle(self, area):
        """
        :type area: int
        :rtype: List[int]
        """
        import math
        L=int(math.ceil(math.sqrt(area)))
        while L<area:
            print(L,area//L)
            if area%L==0:
                return [L,area//L] if L>area//L else [area//L,L]
            L+=1
            
        return [L,area//L]  if L>area//L else [area//L,L] 
        
if __name__ == "__main__":
    print(Solution().constructRectangle(8))         

#496. Next Greater Element I   
class Solution(object):
    def nextGreaterElement(self, findNums, nums):
        """
        :type findNums: List[int]
        :type nums: List[int]
        :rtype: List[int]
        """

        d={}
        st=[]
        
        ans=[]
        
        
        for num in nums:
            while len(st)>0 and st[-1]<num:
                 d[st.pop()]=num
            st.append(num)
            
        
        for x in findNums:
            ans.append(d.get(x,-1))
            
        return ans
        
        
#500. Keyboard Row        
class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """

        a=set('qwertyuiop')
        b=set('asdfghjkl')
        c=set('zxcvbnm')
        
        ans=[]

        for word in words:
            t=set(word.lower())
            if a&t==t:
                ans.append(word)
            if b&t==t:
                ans.append(word)
            if c&t==t:
                ans.append(word)
                
        return ans
        
        
#501. Find Mode in Binary Search Tree        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        from collections import Counter
        counter=Counter()
        
        counter=self.tranverse(root,counter)
        maxn=max(counter.values())
        return [e for e,v in counter.items() if v== maxn ]
        
    def tranverse(self,root,counter):
        
        if not root:
            return 
        
        counter[root.val]+=1
        self.tranverse(root.left,counter)
        self.tranverse(root.right,counter)
        
        return counter
        
        
        
class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans=[]
        pre=None
        cnt=1
        mx=0
       
        pre,cnt,mx,ans=self.inorder(root,pre,cnt,mx,ans)
        
        return ans
        
    def inorder(self,node,pre,cnt,mx,ans):
        if not node:
            return pre,cnt,mx,ans
        
        pre,cnt,mx,ans=self.inorder(node.left,pre,cnt,mx,ans)
        
        
        if pre:
           
           if node.val==pre.val:
               cnt=1+cnt
           else:
               cnt=1
        if cnt>=mx:
            if cnt>mx:
                #print(cnt,mx,ans)
                del ans[:]
            ans.append(node.val)
            
            mx=cnt
            #print(cnt,mx,ans)
        pre=node
        return self.inorder(node.right,pre,cnt,mx,ans)
       
        
                
#504. Base 7       
class Solution(object):
    def convertToBase7(self, num):
        """
        :type num: int
        :rtype: str
        """

        if num <0:
            return '-'+self.convertToBase7(-num)
        if num<7:
            return str(num%7)
        
        return self.convertToBase7(num//7)+str(num%7)
        
        
#506. Relative Ranks        
class Solution(object):
    def findRelativeRanks(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        numsSort=sorted(nums)[::-1]
        rank=["Gold Medal", "Silver Medal", "Bronze Medal"] +list(map(str,list(range(4,len(nums)+1))))
        print(numsSort,rank)
        return      list(map(dict(zip(numsSort,rank)).get,nums))
        
if __name__ == "__main__":
    print(Solution().findRelativeRanks([5,4,3,2,1]))         

#507. Perfect Number
class Solution(object):
    def checkPerfectNumber(self, num):
        """
        :type num: int
        :rtype: bool
        """
        import math
               
        
        if num<2:
            return False
        
        divisor=[]
        
        for i in range(2,int(math.ceil(math.sqrt(num)))):
            if num%i==0:
               divisor.append(i) 
               divisor.append(num/i)
               
        print(divisor)
        if sum(divisor)==num-1:
            return True
        else:
            return False
        
if __name__ == "__main__":
    print(Solution().checkPerfectNumber(5))         
        
        
#520. Detect Capital        
class Solution(object):
    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        if not word:
            return False
        
        
        if word.isupper():
            return True
        elif word.islower():
            return True
        elif word[0].isupper()  and word[1:].islower():
            return True
        
        else:
            return False
        
if __name__ == "__main__":
    print(Solution().detectCapitalUse("Leetcode"))        
        
#521. Longest Uncommon Subsequence I      
class Solution(object):
    def findLUSlength(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """

        if a==b:
            return -1
        else:
            return max(len(a),len(b))
        
        
#530. Minimum Absolute Difference in BST        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        ans=float("inf")
        prev=None
        
        prev,ans=self.inorder(root,prev,ans)
        
        return ans
        


    def inorder(self,node,prev,ans):
        if not node:
            return prev,ans
        
        prev,ans=self.inorder(node.left,prev,ans)
        
        if prev:
            absd=abs(node.val-prev.val)
            if absd<ans:
                ans=absd
        prev=node    
        
        
        
        return self.inorder(node.right,prev,ans)
        
#532. K-diff Pairs in an Array        
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        from collections import Counter
        counter=Counter(nums)    
        res=0
        for i in counter:
            if k>0 and i+k in counter or (k==0 and counter[i]>1):
               res+=1
        return res
if __name__ == "__main__":
    print(Solution().findPairs([3, 1, 4, 1, 5],1))            
        
        
        
#538. Convert BST to Greater Tree        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        
        self.sumR=0
        
        
        
        def tranverse(node):
            if not node:
                return 
            
            tranverse(node.right)
            self.sumR+=node.val
            node.val=self.sumR
                
            tranverse(node.left)
            
        
        tranverse(root)
        return root
        
#541. Reverse String II        
class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        
        if not s:
            return ''
        
        if k==0:
            return s
        res=''
        return self.reversehelp(s,k,res)            
            
    def reversehelp(self,subStr,k,res):
        
        if not subStr:
            return ''
        n=len(subStr)
        if n < k:
            res+=subStr[::-1]
        elif n >= k and n<2*k:
            
            res+=subStr[k-1::-1]+subStr[k:]
        else:
            res=subStr[k-1::-1]+subStr[k:2*k]+self.reversehelp(subStr[2*k:],k,res)
        return res
if __name__ == "__main__":
    print(Solution().reverseStr("ab",2))         
        
#543. Diameter of Binary Tree        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root:
            return 0
        
        self.ans=0
        def depth(node):
            if not node:
                return 0
            
            left,right=depth(node.left),depth(node.right)
            self.ans=max(self.ans,left+right)
            return 1+max(left,right)
            
        depth(root)
        return self.ans
        
#551. Student Attendance Record I        
class Solution(object):
    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        from collections import Counter
        
        if not s:
            return True
        
        sc=Counter(s)
        
        if sc['A']>1:
            return False
        if sc['L']>2:
            
            for i  in range(len(s)-2):
                if s[i]=='L' and s[i+1]=='L' and s[i+2]=='L':
                    return False
        return True
        
#557. Reverse Words in a String III        
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        l=[]
        for word in s.split():
            l.append(word[::-1])
        return " ".join(l)    
            
        
#561. Array Partition I        
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return  sum(sorted(nums)[::2])
              
#563. Binary Tree Tilt 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        Tilt=0
        def _sum(node):
            if not node:
                return 0
            left,right=_sum(node.left),_sum(node.right)
            Tilt+=abs(left-right)
            return left+right+node.val
        
        _sum(root)
        return Tilt
        
#566. Reshape the Matrix        
class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        
        if not nums:
            return []
        
        n=len(nums)
        m=len(nums[0])
        if n*m!=r*c:
            return nums
        numsFlat=[]
        
        for num in nums:
            numsFlat+=num
            
        ans=[]
        for i in range(r):
            ans.append(numsFlat[i*c:c*(i+1)])
        return ans
if __name__ == "__main__":
    print(Solution().matrixReshape([[1,2], [3,4]],4,1))            
        
        
#572. Subtree of Another Tree        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        if not s and not t:
            return True
        if  not s or not t:
            return False 
        
        return self.isSame(s,t)   or  self.isSubtree(s.left,t) or self.isSubtree(s.right,t)
        
    
    
    
    def isSame(self,node1,node2):
        if not node1 and not node2:
            return True
        if  not node1 or not node2:
            return False
        return node1.val==node2.val and self.isSame(node1.left,node2.left) and self.isSame(node1.right,node2.right)
        
        
    
#575. Distribute Candies        
class Solution(object):
    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        from collections import Counter
        counter=Counter(candies)  
        t=sorted(list(counter.values()))
        
        if len(t)>=len(candies)/2:
            return int(len(candies)/2)
        else:
            return len(t)
        
        
            
        
if __name__ == "__main__":
    print(Solution().distributeCandies([1,1,2,2,3,3]))         
        
#581. Shortest Unsorted Continuous Subarray        
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        sorted_nums=sorted(nums)
        #print(sorted_nums)
        
        if len(nums)==1:
            return 0
        minindex=-1
        maxindex=-1
        for i in range(len(nums)):
            if nums[i]!=sorted_nums[i]:
               minindex=i
               print(minindex)
               break
            
        for i in range(len(nums)):    
            if nums[len(nums)-1-i]!=sorted_nums[len(nums)-1-i]:
               maxindex=len(nums)-1-i
               break
        if maxindex==-1:
            return 0
        return maxindex-minindex+1
if __name__ == "__main__":
    print(Solution().findUnsortedSubarray([2, 6, 4, 8, 10, 9, 15]))        
        
#594. Longest Harmonious Subsequence        
class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if not nums or len(nums)==1:
            return 0
        
        from collections import Counter
        counter=Counter(nums)  

        numsort=sorted(nums)  
        maxHS=0
        for i in range(len(numsort)-1):
            if numsort[i]- numsort[i+1]==-1:
                HS=counter[numsort[i]]+counter[numsort[i+1]]
                if HS>maxHS:
                   maxHS=HS
        return maxHS
if __name__ == "__main__":
    print(Solution().findLHS([1,3,2,2,5,2,3,7]))            
            
            
            
#595. Big Countries            
# Write your MySQL query statement below
#select name, population, area from World
#where population >25000000 or area > 3000000            
#            
            
#596. Classes More Than 5 Students            
## Write your MySQL query statement below
#select class from courses
#group by class
#
#having count(distinct student)>=5            
            
#598. Range Addition II            
class Solution(object):
    def maxCount(self, m, n, ops):
        """
        :type m: int
        :type n: int
        :type ops: List[List[int]]
        :rtype: int
        """    
        if m==0 and n==0:
            return 0
        if not ops:
            return m*n
        
        
        minx=float("inf")
        miny=float("inf")
        
        for [x,y] in ops:
            if x < minx:
                minx=x
            if y < miny:
                miny=y
                
        return min(minx,m)*min(miny,n)
if __name__ == "__main__":
    print(Solution().maxCount(3,3,[[2,2],[3,3]]))                  
            
#599. Minimum Index Sum of Two Lists            
class Solution(object):
    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        
        d1={}
        d2={}
        d3={}
        
        
        for i, item in enumerate(list1):
            d1[item]=i
        for i, item in enumerate(list2):
            d2[item]=i    
        
        common =[ val for val in list1 if val in list2]
        print(common)
        
        minindex=float("inf")
        for item in common:
            if minindex > d1[item]+ d2[item]:
               minindex = d1[item]+ d2[item]
            print(d1[item],d2[item])
            d3[item] =d1[item]+ d2[item]
            print(d3)
        return [item for item,k in d3.items() if k==minindex]
list1=["Shogun", "Tapioca Express", "Burger King", "KFC"]
list2 =["KFC", "Shogun", "Burger King"]   
if __name__ == "__main__":
    print(Solution().findRestaurant( list1, list2))         
        
#605. Can Place Flowers            
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        if not flowerbed:
            return False
        
        
        count=0
        i=0
        while i<len(flowerbed):
            if flowerbed[i]==0 and (i==0 or flowerbed[i-1]==0) and (i==len(flowerbed)-1 or flowerbed[i+1]==0  ):
               flowerbed[i]=1
               count+=1
               print(flowerbed,count)
            i+=1
        return count >=n
            
if __name__ == "__main__":
    print(Solution().canPlaceFlowers( [1,0,0,0,0,1], 2))             
            
            
#606. Construct String from Binary Tree            
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """            
        if not t:
            return ''
        result=str(t.val)
        left=self.tree2str(t.left)
        right=self.tree2str(t.right)
        
        if not t.left  and not t.right:
            return result
        if not left:
            return result+'()'+'('+right+')'
        if not right:
            return result+'('+left+')'
        if t.left  and t.right:
            return result+'('+left+')'+'('+right+')'
        
#617. Merge Two Binary Trees        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1:
            return t2
        if not t2:
            return t1
        t1.val=t1.val+t2.val
        t1.left=self.mergeTrees( t1.left, t2.left)
        t1.right=self.mergeTrees( t1.right, t2.right)
        
        return t1
        
#620. Not Boring Movies        
# Write your MySQL query statement below

#select * from cinema
#where id % 2 =1 and description not in ( 'boring' )
#order by rating desc        
        
#627. Swap Salary        
# Write your MySQL query statement below
#update salary
#set 
#sex=  case sex
#when 'f' then 'm'
#when 'm' then 'f'
#end;        
#628. Maximum Product of Three Numbers        
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if not list:
            return None
        s_nums=sorted(nums)
        positive=s_nums[-1]*s_nums[-2]*s_nums[-3]
        mix=s_nums[0]*s_nums[1]*s_nums[-1]
        return positive if positive >mix else mix
        
#633. Sum of Square Numbers        
class Solution(object):
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        import math
        
        hashtable={}
        
        for i in range(int(math.ceil(math.sqrt(c)))+1):
            if c-i*i in hashtable:
                return True
            elif c==2*i*i:
                 return True
            else:
                hashtable[i*i]=i
         
        return False
        
#637. Average of Levels in Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        ans=[]
        if not root:
            return []
        
        from collections  import deque
        
        dq=deque()
        dq.append(root)
        
        while  dq:
            sm=0
            count=0
            temp=deque()
            while  dq:
                node=dq.popleft()
                sm+=node.val
                count+=1
                if  node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            dq=temp
            ans.append(sm*1.0/count)
        return ans
        
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        info=[]
        def dfs(node,depth=0):
            if node:
                if len(info)<=depth:
                   info.append([0,0])
                
                info[depth][0]+=node.val
                info[depth][1]+=1
                dfs(node.left,depth+1)
                dfs(node.right,depth+1)
        
        dfs(root)
        return [i*1.0/v for [i,v] in info]


#643. Maximum Average Subarray I
class Solution(object):
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        if len(nums)<=k:
            return 1.0*sum(nums)/len(nums)
        moving=sum(nums[0:k])
        maxv=moving
        for i in range(k,len(nums)):
            moving=moving-nums[i-k]+nums[i]
            if  moving>maxv:
                maxv=moving
        return maxv*1.0/k


#645. Set Mismatch
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []
        d={}
        sumn=0
        sumi=0
        for i, item in enumerate(nums):
            print(d,i,item)
            if item not in d:
                d[item]=1
            else:
                dup=item
            sumn+=item
            sumi+=i+1
        return [dup,sumi-sumn+dup]
if __name__ == "__main__":
    print(Solution().findErrorNums( [1,2,2,4]))               
            

#653. Two Sum IV - Input is a BST
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        if not root:
            return False
    
        bfs,s=[root],set()
        
        for i in bfs:
            if k-i.val in s:
                return True
            s.add(i.val)
            if i.left:
               bfs.append(i.left)
            if i.right:
               bfs.append(i.right)
        return False

#657. Judge Route Circle
class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        if not moves:
            return True
        
        origin=[0,0]
        
        for move in moves:
          if move=='R':
             origin[0] +=1
          if move=='L':
             origin[0] +=-1
          if move=='U':
             origin[1] +=1
          if move=='D':
             origin[1] +=-1

        return origin==[0,0]
        
#661. Image Smoother       
class Solution(object):
    def imageSmoother(self, M):
        """
        :type M: List[List[int]]
        :rtype: List[List[int]]
        """
        from copy import deepcopy
        
        if not M:
            return [[]]
        
        x_len=len(M)
        y_len=len(M[0])
        Mcopy=deepcopy(M)
        for x in range(x_len):
            for y in range(y_len):
                neighbors=[M[_x][_y] for _x in (x-1,x,x+1)  for _y in (y-1,y,y+1)  if  0 <=_x <x_len and 0 <=_y <y_len  ] 
                Mcopy[x][y]=sum(neighbors)//len(neighbors)
        return Mcopy


#665. Non-decreasing Array
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums:
            return True
        
        if len(nums)==1:
            return True
        p=None
        for i in range(len(nums)-1):
            if nums[i]>nums[i+1]:
                if p is not None:
                    return False
                p=i
        return  p is None or p==0 or p==len(nums)-2 or nums[p-1]<=nums[p+1]  or nums[p]<=nums[p+2]
            
             
if __name__ == "__main__":
    print(Solution().checkPossibility( [3,4,2,3]))                  
        
#669. Trim a Binary Search Tree        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        def trim(node):
            if not node:
                return None
            elif node.val>R:
                return trim(node.left)
            elif node.val<L:
                return trim(node.right)
            else:
                node.left=trim(node.left)
                node.right=trim(node.right)
                return node
        return trim(root)

#671. Second Minimum Node In a Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        listnode=[root]
        setNode=set()
        
        
        for node in listnode:
            if node.val not in setNode:
                setNode.add(node.val)
            if node.left:
               listnode.append(node.left)
               listnode.append(node.right)
        n=sorted(list(setNode))
        
        return -1 if len(n) <2 else n[1]
        
#674. Longest Continuous Increasing Subsequence        
class Solution(object):
    def findLengthOfLCIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        maxL=0
        countL=1
        for i in range(len(nums)-1):
            if nums[i]<nums[i+1]:
               countL+=1
            else:
                if countL>maxL:
                   maxL=countL
                countL=1
            
        return max(maxL ,countL)
if __name__ == "__main__":
    print(Solution().findLengthOfLCIS( [2,2,2,2,2]))    


#680. Valid Palindrome II
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        if len(s)<3:
            return True
            
        if self.isPalindrome(s):
            return True
        
        l,r=0,len(s)-1
        
        
        while l<r:
                if s[l]==s[r]:
                    l+=1
                    r-=1
                else:
                    return self.isPalindrome(s[:l]+s[l+1:]) or self.isPalindrome(s[:r]+s[r+1:])
                
       
        
    def isPalindrome(self, string):
            
            if len(string)<2:
                return True
            
            l,r=0,len(string)-1
            
            while l<r:
                if string[l]==string[r]:
                    l+=1
                    r-=1
                else:
                    return False
            return True
                
                
                
if __name__ == "__main__":
    print(Solution().validPalindrome( "abac"))    

                
                
#682. Baseball Game                
class Solution(object):
    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """                
                
        if not ops:
            return 0
 

         
        l=[]        
        for op in ops:
           if op=='C':
              del l[-1]
           elif op=='+':
               l.append(l[-1]+l[-2])
           elif op=='D':
               l.append(l[-1]*2)
           else:
               l.append(int(op))
        return sum(l)
                
if __name__ == "__main__":
    print(Solution().calPoints( ["5","-2","4","C","D","9","+","+"]))   
              
                
#686. Repeated String Match                
class Solution(object):
    def repeatedStringMatch(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: int
        """                
        q=-(-len(B)//len(A))
        for i in range(2):
            if B in A*(q+i):
                return q+i
        return -1
                
#687. Longest Univalue Path                
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """   
        self.ans=0
        def tranverse(node):
            if not node:
                return 0
            left_len=tranverse(node.left)
            right_len=tranverse(node.right)
            left=left_len+1 if node.left and node.left.val==node.val else 0
            right=right_len+1 if node.right and node.right.val==node.val else 0 
            
            self.ans=max(self.ans,left+right)
            return max(left,right)
        tranverse(root)
        return self.ans
                
#690. Employee Importance                
"""
# Employee info
class Employee(object):
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates
"""
class Solution(object):
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        if not employees:
            return 0
        
        emap={e.id:e for e in employees}
        def DFS(eid):
            employee=emap[eid]
            return employee.importance+sum(DFS(eid) for eid in employee.subordinates)
            
        return DFS(id)  
            
            
#693. Binary Number with Alternating Bits            
class Solution(object):
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        n,cur=divmod(n,2)
        
        while n:
            if n&1==cur:
                return False
            n,cur=divmod(n,2)
        return True    
if __name__ == "__main__":
    print(Solution().hasAlternatingBits( 7))   
                          
        
#695. Max Area of Island                
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        row=len(grid)
        col=len(grid[0]) 
        seen=()
        
        def dfs(r,c):
            if not (  0<=r<row and 0<=c<col  and (r,c)  not in seen and  grid[r][c]==1):
                return 0
            seen.add((r,c))
            return 1+dfs(r-1,c)+dfs(r+1,c)+dfs(r,c-1)+dfs(r,c+1)
       
        
        
        return max([dfs[r,c] for r in range(row) for c in range(col)])
        
        
#696. Count Binary Substrings        
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        if not s:
            return 0
        import  itertools  
        group= [len(list(g)) for    _ , g in itertools.groupby(s)]
        return sum(min(a,b)  for a,b in zip(group,group[1:]))
        
        
#697. Degree of an Array        
class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        from collections import Counter
        counter=Counter(nums)
        maxL=0
        maxk=[]
        for k,item in counter.items():
            if item>maxL:
               maxL=item
               
        for k,item in counter.items():
            if item==maxL:
               maxk.append(k)  
         
        
        d={}
        for i,item in enumerate(nums):
            if item in maxk:
                if item in d:
                    d[item].append(i)
                else:
                    d[item]=[i]
                
        return min([x[-1]-x[0] for k,x in d.items()])        

#717. 1-bit and 2-bit Characters               
class Solution(object):
    def isOneBitCharacter(self, bits):
        """
        :type bits: List[int]
        :rtype: bool
        """
        if not bits:
           return   
        if len(bits) ==1:
           return  True
        if len(bits)==2:
           if bits==[0,0]:
               return True
           else:
               return False
        
        if bits[0]==0:
            return self.isOneBitCharacter(bits[1:])
        else:
            return self.isOneBitCharacter(bits[2:])
            
        
#720. Longest Word in Dictionary        
class Solution(object):
    def longestWord(self, words):
        """
        :type words: List[str]
        :rtype: str
        """        
        if not words:
            return []
        
        ans=''
        wordset=set(words)
        for word in words:
            if len(word)>len(ans) or ( len(word)==len(ans) and word < ans):
                if all( word[:k] in wordset for k in range(1,len(word) )):
                    ans=word
        return ans
        
#724. Find Pivot Index        
class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """        
        if not nums :
            return -1
        
        if len(nums)==1:
            return -1
        
        sumN=sum(nums)
        sumCur=0
        
        if sumN-nums[0]==0:
            return 0
        
        
        
        for i in range(len(nums)-1):
            sumCur+=nums[i]
            if sumCur==sumN-nums[i+1]-sumCur:
                return i+1
            
        if sumN-nums[-1]==0:
            return len(nums) -1
        
        return -1
        
#728. Self Dividing Numbers            
class Solution(object):
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        
        ans=[]
            
        def isdivisible(num):
            digits=[]
            temp=num
            while num:
                  digits.append(num%10)
                  num=num//10
            
            for digit in digits:
                if not digit:
                   return False
                if temp%digit!=0:
                    return False
            return True
            
        for i in range(left,right+1):
             if isdivisible(i):
                 ans.append(i)
        return ans
             
if __name__ == "__main__":
    print(Solution().selfDividingNumbers( 1,22)) 


#733. Flood Fill
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        if not image:
            return [[]]
        
        r=len(image)
        c=len(image[0])
        seen=set()
        starting=image[sr][sc]
        
        def dfs(image, sr, sc,newColor):
            if not (0<=sr<r  and 0<=sc<c and image[sr][sc]==starting and (sr,sc) not in seen):
               return      
             
            
            image[sr][sc]=newColor            
            print((sr,sc),image[sr][sc],newColor)
            seen.add((sr,sc))
            dfs(image, sr-1, sc, newColor)
            dfs(image, sr+1, sc, newColor)
            dfs(image, sr, sc-1, newColor)
            dfs(image, sr, sc+1, newColor)
            
            
       
        dfs(image, sr, sc,newColor)
        return image

if __name__ == "__main__":
    print(Solution().floodFill( [[0,0,0],[0,0,0],[0,0,0]],0,0,2)) 




#734. Sentence Similarity
class Solution(object):
    def areSentencesSimilar(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        
        if not words1 and not words2:
            return True
        
        if len( words1)!=len( words2):
            return False
        
        for w1, w2 in zip(words1,words2):
            if not ([w1, w2] in pairs or [w2, w1]  in pairs or w1==w2):
                return False
        return True
            
#744. Find Smallest Letter Greater Than Target
class Solution(object):
    def nextGreatestLetter(self, letters, target):
        """
        :type letters: List[str]
        :type target: str
        :rtype: str
        """
        res='z'
        for letter in letters:
            if ord(letter) > ord(target) and ord(letter) < ord(res):
                print(ord(letter) ,ord(target))
                res=letter
                return res
        return letters[0]
if __name__ == "__main__":
    print(Solution().nextGreatestLetter(["c","f","j"],'j')) 


#746. Min Cost Climbing Stairs
class Solution:
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        res=[0 for _ in range(len(cost))]
        
        res[0]=cost[0]
        res[1]=cost[1]
        
        for i in range(2,len(cost)):
            res[i]=min(cost[i]+res[i-2],res[i-1]+cost[i])
            
        print(cost)
        print(res)   
        return res

if __name__ == "__main__":
    print(Solution().minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])) 


#747. Largest Number At Least Twice of Others
class Solution:
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        if len(nums) ==1:
            return -1
        
        small =0
        big=0
        for i in range(len(nums)):
            if nums[i]>big:
                small=big
                big=nums[i]
                bigi=i
            elif nums[i]>small:
                small=nums[i]
          
        if small*2<=big:
            return bigi
        else:
            return -1
if __name__ == "__main__":
    print(Solution().dominantIndex([1, 2, 3, 4])) 
        
#760. Find Anagram Mappings        
class Solution:
    def anagramMappings(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        
        from collections import defaultdict 
        d=defaultdict(list)
        
        
        for i,num in enumerate(B):
            d[num]+=[i]
        
        res=[]
        for num in A:
            res.append(d[num].pop())
        return res
        

A = [12, 28, 46, 32, 50]
B = [50, 12, 32, 46, 28]
if __name__ == "__main__":
    print(Solution().anagramMappings(A,B))             
            
            
#762. Prime Number of Set Bits in Binary Representation            
class Solution:
    def countPrimeSetBits(self, L, R):
        """
        :type L: int
        :type R: int
        :rtype: int
        """
        res=0
        
        def isPrime(x):
            if x<2:
                return False
            if x==2 or x==3:
                return True
            
            for i in range(2,int(x**0.5)+1):
                if not x%i:
                    return False
            return True
                
        
        
        for num in range(L,R+1):
            
            if isPrime(bin(num).count('1')):
                res+=1
        return res
if __name__ == "__main__":
    print(Solution().countPrimeSetBits(10,15))            

#766. Toeplitz Matrix        
class Solution:
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        if not matrix:
            return True
        if not matrix[0]:
            return True
        def isvalid(x,y):
            if -1<x<len(matrix) and -1<y<len(matrix[0]):
                return True
            else:
                return False
            
        def isequal(x,y):
            while isvalid(x+1,y+1):
                print(matrix[x][y],matrix[x+1][y+1])
                if matrix[x][y]!=matrix[x+1][y+1]:
                    return False
                x=x+1
                y=y+1
            return True
        
        for i in range(len(matrix)):
            if not isequal(i,0):
                return False
        for j in range(len(matrix[0])):
            if not isequal(0,j):
                return False
        return True
matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
matrix = [[1,2],[2,2]]
if __name__ == "__main__":
    print(Solution().isToeplitzMatrix(matrix))            
        
#771. Jewels and Stones         
class Solution:
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        if not S or not J:
            return 0
        
        from collections import Counter
        count=Counter(S)
        
        res=0
        for j in J:
            res+=count[j]
        return res
J = "aA"
S = "aAAbbbb" 
J = "z"
S = "ZZ"
if __name__ == "__main__":
    print(Solution(). numJewelsInStones(J, S))                 
             
        
        
        
        
        
        
        
        
               
        
        
        
        