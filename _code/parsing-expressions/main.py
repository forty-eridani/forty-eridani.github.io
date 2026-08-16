from enum import Enum

class TokenType(Enum):
    PLUS = 0
    HYPHEN = 1
    STAR = 2
    SLASH = 3
    NUMBER = 4
    OPEN_PAREN = 5
    CLOSED_PAREN = 6
    END = 7
    
def tokenize(src):
  cur_index = 0
  tokens = []

  while cur_index < len(src):
      if src[cur_index] == '+':
          tokens.append((TokenType.PLUS, 0))
          cur_index += 1
      elif src[cur_index] == '-':
          tokens.append((TokenType.HYPHEN, 0))
          cur_index += 1
      elif src[cur_index] == '*':
          tokens.append((TokenType.STAR, 0))
          cur_index += 1
      elif src[cur_index] == '/':
          tokens.append((TokenType.SLASH, 0))
          cur_index += 1
      elif src[cur_index] == '(':
          tokens.append((TokenType.OPEN_PAREN, 0))
          cur_index += 1
      elif src[cur_index] == ')':
          tokens.append((TokenType.CLOSED_PAREN, 0))
          cur_index += 1
      elif src[cur_index].isnumeric():
        start = cur_index

        while cur_index < len(src) and src[cur_index].isnumeric():
          cur_index += 1

        tokens.append((TokenType.NUMBER, int(src[start:cur_index])))

      elif src[cur_index] == ' ':
        cur_index += 1
      else:
        raise Exception(f"Unknown character '{src[cur_index]}'.")

  return tokens
  
def evaluate(tokens):
  stack = []

  for token in tokens:
    if token[0] == TokenType.NUMBER:
      # Appending is pushing to python lists
      stack.append(token[1])
    elif token[0] == TokenType.PLUS:
      # The left operand is the first one pushed so is the last one popped
      b = stack.pop()
      a = stack.pop()
      stack.append(a + b)
    elif token[0] == TokenType.HYPHEN:
      b = stack.pop()
      a = stack.pop()
      stack.append(a - b)
    elif token[0] == TokenType.STAR:
      b = stack.pop()
      a = stack.pop()
      stack.append(a * b)
    elif token[0] == TokenType.SLASH:
      b = stack.pop()
      a = stack.pop()
      stack.append(a / b)
  
  return stack.pop()
  
def parse_add_sub(index, rpn, src):
  parse_mult_div(index, rpn, src)
  
  while index[0] < len(src) and (src[index[0]][0] == TokenType.PLUS or src[index[0]][0] == TokenType.HYPHEN):
    op = src[index[0]]
    index[0] += 1

    # Parsing the right operand
    parse_mult_div(index, rpn, src)

    rpn.append(op)

def parse_mult_div(index, rpn, src):
  parse_primary(index, rpn, src)
  
  while index[0] < len(src) and (src[index[0]][0] == TokenType.STAR or src[index[0]][0] == TokenType.SLASH):
    op = src[index[0]]
    index[0] += 1

    # Parsing the right operand
    parse_primary(index, rpn, src)

    rpn.append(op)

def parse_primary(index, rpn, src):
  if src[index[0]][0] == TokenType.NUMBER:
    rpn.append(src[index[0]])
    index[0] += 1
  elif src[index[0]][0] == TokenType.OPEN_PAREN:
    index[0] += 1
    parse_add_sub(index, rpn, src)

    if src[index[0]][0] != TokenType.CLOSED_PAREN:
      raise Exception("Expected closing parenthesis after group.")

    index[0] += 1
  else:
    raise Exception(f"Unexpected token '{src[index[0]][0]}'")
  
def parse(tokens):
  rpn = []
  # Python doesn't allow numbers to be passed by reference, but the index 
  # must be modified by each call. This is a little workaround to pass a 
  # list which is essentially passed by reference
  index = [0]

  parse_add_sub(index, rpn, tokens)

  return rpn
  
  
expression = "6 * (7- 4) + 3"
tokens = tokenize(expression)
parsed_tokens = parse(tokens)
value = evaluate(parsed_tokens)
print(f"The value of {expression} is {value}.")