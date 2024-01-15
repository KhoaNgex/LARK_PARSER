from typing import List

from lark import Lark, Transformer, Token, v_args, ast_utils
from lexererr import *
from lark.exceptions import UnexpectedCharacters, UnexpectedToken

import sys

this_module = sys.modules[__name__]

all_grammar = r"""
        start: decl+
        
        decl: vardecl | funcdecl 
        
        assignment_statement: scalar_variable ASSIGN expression SEMI
        scalar_variable: IDENTIFIERS | index_element

        if_statement: IF LP expression RP statement (ELSE statement)?

        for_statement: FOR LP init_for COMMA expression COMMA expression RP statement
        init_for: scalar_variable ASSIGN expression

        while_statement: WHILE LP expression RP statement

        dowhile_statement: DO block_statement WHILE LP expression RP SEMI

        break_statement: BREAK SEMI

        continue_statement: CONTINUE SEMI

        return_statement: RETURN expression? SEMI

        call_statement: IDENTIFIERS LP arglist RP SEMI
        function_call: IDENTIFIERS LP arglist RP
        arglist: (expression (COMMA expression)*)?

        block_statement: LB statement_vardecl_list RB

        statement: assignment_statement
            | if_statement
            | for_statement
            | while_statement
            | dowhile_statement
            | break_statement
            | continue_statement
            | return_statement
            | call_statement
            | block_statement
        statement_vardecl_list: statement_vardecl*
        statement_vardecl: statement | vardecl
    
        expression: expression1 CONCATOP expression1 | expression1
        expression1: expression2 EQUALOP expression2
            | expression2 INEQUALOP expression2
            | expression2 LESSOP expression2
            | expression2 GREATEROP expression2
            | expression2 LESSEQUALOP expression2
            | expression2 GREATEREQUALOP expression2
            | expression2
        expression2: expression2 ANDOP expression3
            | expression2 OROP expression3
            | expression3
        expression3: expression3 ADDOP expression4
            | expression3 SUBOP expression4
            | expression4
        expression4: expression4 MULOP expression5
            | expression4 DIVOP expression5
            | expression4 MODOP expression5
            | expression5
        expression5: NOTOP expression5 | expression6
        expression6: SUBOP expression6 | expression7
        expression7: LP expression RP
            | TRUE 
            | FALSE
            | literals
            | function_call
            | scalar_variable
        index_element: IDENTIFIERS LSB expression_list RSB

        vardecl: vardecl_recur SEMI
        vardecl_recur: IDENTIFIERS (COMMA IDENTIFIERS)* COLON variable_type (ASSIGN expression (COMMA expression)*)?

        paramdecl: [INHERIT] [OUT] IDENTIFIERS COLON variable_type
        paramdecl_list: (paramdecl (COMMA paramdecl)*)?

        funcdecl: func_prototype func_body
        func_prototype: not_inherit_func_prototype | inherit_func_prototype
        not_inherit_func_prototype: IDENTIFIERS COLON FUNCTION function_type LP paramdecl_list RP
        inherit_func_prototype: not_inherit_func_prototype INHERIT IDENTIFIERS
        func_body: block_statement
        
        literals: atomic_type_value | indexed_array
        atomic_type: INTEGER | STRING | BOOLEAN | FLOAT
        array_dimension: INTLIT (COMMA INTLIT)*
        function_type: atomic_type | arra_type | VOID | AUTO
        variable_type: atomic_type | arra_type | AUTO
        arra_type: ARRAY LSB array_dimension RSB OF atomic_type

        atomic_type_value: INTLIT | FLOATLIT | STRINGLIT 

        expression_list: expression (COMMA expression)*
        indexed_array: LB (expression_list)? RB
        
        FLOATLIT: /[1-9](_?[0-9])*\.[0-9]*/ 
                | /0\.[0-9]*/ 
                | /[1-9](_?[0-9])*[eE][-+]?[0-9]+/
                | /0[eE][-+]?[0-9]+/
                | /\.[0-9]*[eE][-+]?[0-9]+/
                | /[1-9](_?[0-9])*\.[0-9]*[eE][-+]?[0-9]+/ 
                | /0\.[0-9]*[eE][-+]?[0-9]+/ 
        INTLIT: "0" | /[1-9](_?[0-9])*/
        STRINGLIT: "\"" [CHARACTER*] "\""
        CHARACTER: /[^"\r\n\\]|\\["bfrnt'\\]/
        
        AUTO: "auto"
        BREAK: "break"
        BOOLEAN: "boolean"
        DO: "do"
        ELSE: "else"
        FALSE: "false"
        FLOAT: "float"
        FOR: "for"
        FUNCTION: "function"
        IF: "if"
        INTEGER: "integer"
        RETURN: "return"
        STRING: "string"
        TRUE: "true"
        WHILE: "while"
        VOID: "void"
        OUT: "out"
        CONTINUE: "continue"
        OF: "of"
        INHERIT: "inherit"
        ARRAY: "array"

        ADDOP: "+"
        SUBOP: "-"
        MULOP: "*"
        DIVOP: "/"
        MODOP: "%"
        NOTOP: "!"
        ANDOP: "&&"
        OROP: "||"
        EQUALOP: "=="
        INEQUALOP: "!="
        LESSOP: "<"
        GREATEROP: ">"
        LESSEQUALOP: "<="
        GREATEREQUALOP: ">="
        CONCATOP: "::"

        LP: "("
        RP: ")"
        LSB: "["
        RSB: "]"
        LB: "{"
        RB: "}"

        DOT: "."
        COMMA: ","
        SEMI: ";"
        COLON: ":"
        ASSIGN: "="
        
        IDENTIFIERS: /[A-Za-z_][A-Za-z0-9_]*/
        CCMT: /\/\*(.|\n)*?\*\//
        CPLUSCMT: /\/\/[^\n]*/
        
        ILLEGAL_ESCAPE: "\"" [CHARACTER*] /\\[^bfrnt'\\"].*/
        UNCLOSE_STRING: "\"" [CHARACTER*]
        
        %import common.WS
        %ignore WS
        %ignore CCMT
        %ignore CPLUSCMT
    """


class _MyTransformer(Transformer):
    def INTLIT(self, tokens):
        value = tokens.replace("_", "")
        return Token(tokens.type, value)

    def FLOATLIT(self, tokens):
        value = tokens.replace("_", "")
        return Token(tokens.type, value)

    def STRINGLIT(self, tokens):
        value = tokens[1:-1]
        return Token(tokens.type, value)

    @v_args(inline=True)
    def ILLEGAL_ESCAPE(self, tokens):
        value = tokens[1:]
        if tokens[-2] == '"':
            value = value[:-2] + value[-1]
        raise IllegalEscape(value)

    @v_args(inline=True)
    def UNCLOSE_STRING(self, tokens):
        value = tokens[1:]
        raise UncloseString(value)


### AST Definition
class _Ast(ast_utils.Ast):
    pass


class _Decl(_Ast):
    pass


class _Statement(_Ast):
    pass


class _Expr(_Statement):
    pass


class _Type(_Ast):
    pass


class _AtomicType(_Type):
    pass


# Expressions
class _LHS(_Expr):
    pass


class IntegerType(_AtomicType):
    def __str__(self):
        return self.__class__.__name__


class FloatType(_AtomicType):
    def __str__(self):
        return self.__class__.__name__


class BooleanType(_AtomicType):
    def __str__(self):
        return self.__class__.__name__


class StringType(_AtomicType):
    def __str__(self):
        return self.__class__.__name__


class ArrayType(_Type):
    def __init__(self, dimensions: List[int], typ: _AtomicType):
        self.dimensions = dimensions
        self.typ = typ

    def __str__(self):
        return "ArrayType([{}], {})".format(
            ", ".join([str(dimen) for dimen in self.dimensions]), str(self.typ)
        )


class AutoType(_Type):
    def __str__(self):
        return self.__class__.__name__


class VoidType(_Type):
    def __str__(self):
        return self.__class__.__name__


# DECLARATION
class VarDecl(_Decl):
    def __init__(self, name: str, typ: _Type, init: _Expr or None = None):
        self.name = name
        self.typ = typ
        self.init = init

    def __str__(self):
        return "VarDecl({}, {}{})".format(
            self.name, str(self.typ), ", " + str(self.init) if self.init else ""
        )


# STATEMENT
class BlockStmt(_Statement):
    def __init__(self, body: List[_Statement or VarDecl]):
        self.body = body

    def __str__(self):
        return "BlockStmt([{}])".format(", ".join([str(body) for body in self.body]))


class ParamDecl(_Decl):
    def __init__(self, name: str, typ: _Type, out: bool = False, inherit: bool = False):
        self.name = name
        self.typ = typ
        self.out = out
        self.inherit = inherit

    def __str__(self):
        return "{}{}Param({}, {})".format(
            "Inherit" if self.inherit else "",
            "Out" if self.out else "",
            self.name,
            str(self.typ),
        )


class FuncDecl(_Decl):
    def __init__(
        self,
        name: str,
        return_type: _Type,
        params: List[ParamDecl],
        inherit: str or None,
        body: BlockStmt,
    ):
        self.name = name
        self.return_type = return_type
        self.params = params
        self.inherit = inherit
        self.body = body

    def __str__(self):
        return "FuncDecl({}, {}, [{}], {}, {})".format(
            self.name,
            str(self.return_type),
            ", ".join([str(param) for param in self.params]),
            self.inherit if self.inherit else "None",
            str(self.body),
        )


class AssignStmt(_Statement):
    def __init__(self, lhs: _LHS, rhs: _Expr):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return "AssignStmt({}, {})".format(str(self.lhs), self.rhs)


class IfStmt(_Statement):
    def __init__(
        self, cond: _Expr, tstmt: _Statement, fstmt: _Statement or None = None
    ):
        self.cond = cond
        self.tstmt = tstmt
        self.fstmt = fstmt

    def __str__(self):
        return "IfStmt({}, {}{})".format(
            str(self.cond),
            str(self.tstmt),
            ", " + str(self.fstmt) if self.fstmt else "",
        )


class ForStmt(_Statement):
    def __init__(self, init: AssignStmt, cond: _Expr, upd: _Expr, stmt: _Statement):
        self.init = init
        self.cond = cond
        self.upd = upd
        self.stmt = stmt

    def __str__(self):
        return "ForStmt({}, {}, {}, {})".format(
            str(self.init), str(self.cond), str(self.upd), str(self.stmt)
        )


class WhileStmt(_Statement):
    def __init__(self, cond: _Expr, stmt: _Statement):
        self.cond = cond
        self.stmt = stmt

    def __str__(self):
        return "WhileStmt({}, {})".format(str(self.cond), str(self.stmt))


class DoWhileStmt(_Statement):
    def __init__(self, cond: _Expr, stmt: BlockStmt):
        self.cond = cond
        self.stmt = stmt

    def __str__(self):
        return "DoWhileStmt({}, {})".format(str(self.cond), str(self.stmt))


class BreakStmt(_Statement):
    def __str__(self):
        return "BreakStmt()"


class ContinueStmt(_Statement):
    def __str__(self):
        return "ContinueStmt()"


class ReturnStmt(_Statement):
    def __init__(self, expr: _Expr or None = None):
        self.expr = expr

    def __str__(self):
        return "ReturnStmt({})".format(str(self.expr) if self.expr else "")


class CallStmt(_Statement):
    def __init__(self, name: str, args: List[_Expr]):
        self.name = name
        self.args = args

    def __str__(self):
        return "CallStmt({}, {})".format(
            self.name, ", ".join([str(expr) for expr in self.args])
        )


class IntegerLit(_Expr):
    def __init__(self, val: int):
        self.val = val

    def __str__(self):
        return "IntegerLit({})".format(self.val)


class FloatLit(_Expr):
    def __init__(self, val: float):
        self.val = val

    def __str__(self):
        return "FloatLit({})".format(self.val)


class StringLit(_Expr):
    def __init__(self, val: str):
        self.val = val

    def __str__(self):
        return "StringLit({})".format(self.val)


class BooleanLit(_Expr):
    def __init__(self, val: bool):
        self.val = val

    def __str__(self):
        return "BooleanLit({})".format("True" if self.val else "False")


class ArrayLit(_Expr):
    def __init__(self, explist: List[_Expr]):
        self.explist = explist

    def __str__(self):
        return "ArrayLit([{}])".format(", ".join([str(exp) for exp in self.explist]))


class ArrayCell(_LHS):
    def __init__(self, name: str, cell: List[_Expr]):
        self.name = name
        self.cell = cell

    def __str__(self):
        return "ArrayCell({}, [{}])".format(
            self.name, ", ".join([str(expr) for expr in self.cell])
        )


class Id(_LHS):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return "Id({})".format(self.name)


class BinExpr(_Expr):
    def __init__(self, op: str, left: _Expr, right: _Expr):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        return "BinExpr({}, {}, {})".format(self.op, str(self.left), str(self.right))


class UnExpr(_Expr):
    def __init__(self, op: str, val: _Expr):
        self.op = op
        self.val = val

    def __str__(self):
        return "UnExpr({}, {})".format(self.op, str(self.val))


class FuncCall(_Expr):
    def __init__(self, name: str, args: List[_Expr]):
        self.name = name
        self.args = args

    def __str__(self):
        return "FuncCall({}, [{}])".format(
            self.name, ", ".join([str(expr) for expr in self.args])
        )


class Program(_Ast):
    def __init__(self, decls: List[_Decl]):
        self.decls = decls

    def __str__(self):
        return "Program([\n\t{}\n])".format(
            "\n\t".join([str(decl) for decl in self.decls])
        )


class ToAst(Transformer):
    def start(self, x):
        decl_lst = list()
        for decl in x:
            if isinstance(decl, list):
                decl_lst += decl
            else:
                decl_lst += [decl]
        return Program(decl_lst)

    # decl: vardecl | funcdecl
    def decl(self, x):
        return x[0]

    # vardecl: vardecl_recur SEMI
    def vardecl(self, x):
        return x[0]

    # vardecl_recur: IDENTIFIERS (COMMA IDENTIFIERS)* COLON variable_type (ASSIGN expression (COMMA expression)*)?
    def vardecl_recur(self, x):
        id_val_lst = list()
        exp_lst = list()
        typ = None
        for ele in x:
            if isinstance(ele, Token) and ele.type == "IDENTIFIERS":
                id_val_lst.append(ele.value)
            if isinstance(ele, _Expr):
                exp_lst.append(ele)
            if isinstance(ele, _Type):
                typ = ele
        if len(exp_lst):
            return [
                str(VarDecl(_assign[0], typ, _assign[1]))
                for _assign in zip(id_val_lst, exp_lst)
            ]
        return [str(VarDecl(_id, typ)) for _id in id_val_lst]

    def variable_type(self, x):
        return x[0]

    # assignment_statement: scalar_variable ASSIGN expression SEMI
    def assignment_statement(self, x):
        return AssignStmt(x[0], x[2])

    # scalar_variable: IDENTIFIERS | index_element
    def scalar_variable(self, x):
        return Id(x[0].value) if isinstance(x[0], Token) else x[0]

    # if_statement: IF LP expression RP statement (ELSE statement)?
    def if_statement(self, x):
        return (
            IfStmt(
                x[2],
                x[4],
                x[6],
            )
            if len(x) == 7
            else IfStmt(x[2], x[4])
        )

    # for_statement: FOR LP init_for COMMA expression COMMA expression RP statement
    def for_statement(self, x):
        return ForStmt(
            x[2],
            x[4],
            x[6],
            x[8],
        )

    # init_for: scalar_variable ASSIGN expression
    def init_for(self, x):
        return AssignStmt(x[0], x[-1])

    # while_statement: WHILE LP expression RP statement
    def while_statement(self, x):
        return WhileStmt(x[2], x[-1])

    # dowhile_statement: DO block_statement WHILE LP expression RP SEMI
    def dowhile_statement(self, x):
        return DoWhileStmt(x[-3], x[1])

    # break_statement: BREAK SEMI
    def break_statement(self, x):
        return BreakStmt()

    # continue_statement: CONTINUE SEMI
    def continue_statement(self, x):
        return ContinueStmt()

    # return_statement: RETURN expression? SEMI
    def return_statement(self, x):
        if len(x) == 3:
            return ReturnStmt(x[1])
        return ReturnStmt()

    # call_statement: IDENTIFIERS LP arglist RP SEMI
    def call_statement(self, x):
        return CallStmt(x[0].value, x[2])

    # function_call: IDENTIFIERS LP arglist RP
    def function_call(self, x):
        return FuncCall(x[0].value, x[2])

    # arglist: (expression (COMMA expression)*)?
    def arglist(self, x):
        return [ele for ele in x if isinstance(ele, _Expr)]

    # block_statement: LB statement_vardecl_list RB
    def block_statement(self, x):
        return BlockStmt(x[1])

    # statement: assignment_statement
    #     | if_statement
    #     | for_statement
    #     | while_statement
    #     | dowhile_statement
    #     | break_statement
    #     | continue_statement
    #     | return_statement
    #     | call_statement
    #     | block_statement
    def statement(self, x):
        return x[0]

    # statement_vardecl_list: statement_vardecl*
    def statement_vardecl_list(self, x):
        return [ele for ele in x]

    # statement_vardecl: statement | vardecl
    def statement_vardecl(self, x):
        return x[0]

    # expression: expression1 CONCATOP expression1 | expression1
    def expression(self, x):
        if len(x) == 1:
            return x[0]
        return BinExpr(x[1].value, x[0], x[-1])

    def expression1(self, x):
        if len(x) == 1:
            return x[0]
        return BinExpr(x[1].value, x[0], x[-1])

    def expression2(self, x):
        if len(x) == 1:
            return x[0]
        return BinExpr(x[1].value, x[0], x[-1])

    def expression3(self, x):
        if len(x) == 1:
            return x[0]
        return BinExpr(x[1].value, x[0], x[-1])

    def expression4(self, x):
        if len(x) == 1:
            return x[0]
        return BinExpr(x[1].value, x[0], x[-1])

    def expression5(self, x):
        if len(x) == 1:
            return x[0]
        return UnExpr(x[0].value, x[1])

    # expression6: SUBOP expression6 | expression7
    def expression6(self, x):
        if len(x) == 1:
            return x[0]
        return UnExpr(x[0].value, x[1])

    # expression7: LP expression RP | TRUE | FALSE
    #         | literals
    #         | function_call
    #         | scalar_variable
    def expression7(self, x):
        if len(x) == 3:
            return x[1]
        elif isinstance(x[0], Token) and x[0].type in ["TRUE", "FALSE"]:
            return BooleanLit(x[0].value == "true")
        return x[0]

    # index_element: IDENTIFIERS LSB expression_list RSB
    def index_element(self, x):
        return ArrayCell(x[0].value, x[2])

    # paramdecl: [INHERIT] [OUT] IDENTIFIERS COLON variable_type
    def paramdecl(self, x):
        is_inherit = False
        is_out = False
        id_val = None
        for ele in x:
            if isinstance(ele, Token) and ele.type == "INHERIT":
                is_inherit = True
            if isinstance(ele, Token) and ele.type == "OUT":
                is_out = True
            if isinstance(ele, Token) and ele.type == "IDENTIFIERS":
                id_val = ele.value
        return ParamDecl(
            id_val,
            x[-1],
            is_out,
            is_inherit,
        )

    # paramdecl_list: (paramdecl (COMMA paramdecl)*)?
    def paramdecl_list(self, x):
        return [ele for ele in x if isinstance(ele, _Decl)]

    # funcdecl: func_prototype func_body
    def funcdecl(self, x):
        function_prototype = x[0]
        func_body = x[1]
        return [
            FuncDecl(
                function_prototype["name"],
                function_prototype["type"],
                function_prototype["paramdecl_list"],
                function_prototype["inherit"],
                func_body,
            )
        ]

    # func_prototype: not_inherit_func_prototype | inherit_func_prototype
    def func_prototype(self, x):
        return x[0]

    # not_inherit_func_prototype: IDENTIFIERS COLON FUNCTION function_type LP paramdecl_list RP
    def not_inherit_func_prototype(self, x):
        function_prototype_data = {
            "name": x[0].value,
            "type": x[3],
            "paramdecl_list": x[5],
            "inherit": None,
        }
        return function_prototype_data

    # inherit_func_prototype: not_inherit_func_prototype INHERIT IDENTIFIERS
    def inherit_func_prototype(self, x):
        function_prototype_data = x[0]
        function_prototype_data["inherit"] = x[-1].value
        return function_prototype_data

    # func_body: block_statement
    def func_body(self, x):
        return x[0]

    def literals(self, x):
        return x[0]

    def atomic_type_value(self, x):
        if x[0].type == "INTLIT":
            return IntegerLit(int(x[0].value))
        elif x[0].type == "FLOATLIT":
            float_lit_str = x[0].value
            if float_lit_str.startswith(".e"):
                return FloatLit(0.0)
            return FloatLit(float(float_lit_str))
        elif x[0].type == "STRINGLIT":
            return StringLit(str(x[0].value))

    def atomic_type(self, x):
        if x[0].type == "INTEGER":
            return IntegerType()
        elif x[0].type == "BOOLEAN":
            return BooleanType()
        elif x[0].type == "FLOAT":
            return FloatType()
        elif x[0].type == "STRING":
            return StringType()

    # function_type: atomic_type | arra_type | VOID | AUTO
    def function_type(self, x):
        if isinstance(x[0], Token):
            return VoidType() if x[0].type == "VOID" else AutoType()
        return x[0]

    # variable_type: atomic_type | arra_type | AUTO
    def variable_type(self, x):
        if isinstance(x[0], Token):
            return AutoType()
        return x[0]

    # arra_type: ARRAY LSB array_dimension RSB OF atomic_type
    def arra_type(self, x):
        dimensions = x[2]
        typ = x[-1]
        return ArrayType(dimensions, typ)

    # array_dimension: INTLIT (COMMA INTLIT)*
    def array_dimension(self, x):
        return [int(ele.value) for ele in x if ele.type == "INTLIT"]

    # expression_list: expression (COMMA expression)*
    def expression_list(self, x):
        exp_lst = list()
        for ele in x:
            if isinstance(ele, _Expr):
                exp_lst.append(ele)
        return exp_lst

    # indexed_array: LB (expression_list)? RB
    def indexed_array(self, x):
        return ArrayLit(list()) if len(x) == 2 else ArrayLit(x[1])


MT22_parser = Lark(
    all_grammar,
    start="start",
    transformer=_MyTransformer(),
    parser="lalr",
)

AST_transformer = ast_utils.create_transformer(this_module, ToAst())

if __name__ == "__main__":
    text = """ a:string = "He asked me: \\\"Where is John?\\\""; """
    try:
        parse_tree = MT22_parser.parse(text)
        ## lexer output
        token_list = parse_tree.scan_values(lambda v: isinstance(v, Token))
        print("LEXER'S OUTPUT")
        print("-----------------------------------------------------------------")
        print(",".join([token.value for token in token_list]) + ",<EOF>")
        print("-----------------------------------------------------------------")
        ast_tree = AST_transformer.transform(parse_tree)
        ## parser output
        print("PARSER'S OUTPUT")
        print("-----------------------------------------------------------------")
        print("successful")
        print("-----------------------------------------------------------------")
        ## AST Generation output
        print("AST GEN'S OUTPUT")
        print("-----------------------------------------------------------------")
        print(ast_tree)
        print("-----------------------------------------------------------------")
    except LexerError as e:
        print("LEXER'S OUTPUT")
        print("-----------------------------------------------------------------")
        print(e.message)
        print("-----------------------------------------------------------------")
    except UnexpectedCharacters as e:
        print("LEXER'S OUTPUT")
        print("-----------------------------------------------------------------")
        print(ErrorToken(e.char).message)
        print("-----------------------------------------------------------------")
    except UnexpectedToken as e:
        print("PARSER'S OUTPUT")
        print("-----------------------------------------------------------------")
        print(f"Error on line {e.line} col {e.column}: {e.token}")
        print("-----------------------------------------------------------------")
