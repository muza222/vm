"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""
import asyncio
import builtins
import collections
import dis
import types
import typing as tp
import collections.abc
import operator


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.10/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.value = None
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value: tp.Callable[[int], tp.Any] | None = None
        self.loop_endings: tp.Any = []
        self.current_offset: int = 0
        self.allow_change_offset_in_run: bool = False
        self.offset_map = {instr.offset: instr for instr in dis.get_instructions(frame_code)}
        self.current_offset = 0
        self.block_stack: tp.Any = []
        self.instruction_mapping: tp.Any = {instr.offset: instr for instr in dis.get_instructions(self.code)}
        self.max_offset = (len(self.instruction_mapping) - 1) * 2
        self.offset = 0
        self.next = 0
        self.bytecode_counter = 0

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def topn(self, n: int) -> tp.Any:
        """
         top a number of values from the value stack.
         A list of n values is returned, the deepest value first.
         """
        if n > 0:
            return self.data_stack[-n:]
        else:
            return []

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        instructions = list(dis.get_instructions(self.code))
        while True:
            instruction = instructions[self.bytecode_counter // 2]
            self.bytecode_counter += 2
            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            if instruction.opname == "RETURN_VALUE":
                break

        return self.return_value

    def nop_op(self, arg: str) -> None:
        pass

    def rot_n_op(self, arg: int) -> None:
        self.data_stack[-arg:] = self.data_stack[-1:] + self.data_stack[-arg:-1]

    def rot_two_op(self, arg: int) -> None:
        self.rot_n_op(2)

    def rot_three_op(self, arg: str) -> None:
        self.rot_n_op(3)

    def rot_four_op(self, arg: str) -> None:
        self.rot_n_op(4)

    def dup_top_op(self, arg: str) -> None:
        self.push(self.top())

    def dup_top_two_op(self, arg: int) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1, tos, tos1, tos)

    def unary_positive_op(self, arg: str) -> None:
        top_value = self.top()
        self.data_stack[-1] = top_value

    def unary_negative_op(self, arg: str) -> None:
        top_value = self.top()
        self.data_stack[-1] = -top_value

    def unary_invert_op(self, arg: str) -> None:
        top_value = self.top()
        self.data_stack[-1] = ~top_value

    def unary_not_op(self, arg: str) -> None:
        top_value = self.top()
        self.data_stack[-1] = not top_value

    def get_iter_op(self, arg: str) -> None:
        self.push(iter(self.pop()))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-LOAD_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2829
        """
        # TODO: parse all scopes
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def call_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-CALL_FUNCTION
        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L3496
        """
        arg = self.popn(arg)
        func = self.pop()
        self.push(func(*arg))

    def yield_value_op(self, arg: str) -> None:
        return self.top()

    def yield_from_op(self, tos: str) -> None:
        _ = self.pop()
        raise NotImplementedError

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-LOAD_CONST

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1871
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-RETURN_VALUE

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2436
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-POP_TOP

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1886
        """
        self.pop()

    "Build:"

    def build_tuple_op(self, arg: int) -> None:
        self.push(tuple(self.popn(arg)))

    def build_list_op(self, arg: int) -> None:
        self.push(list(self.popn(arg)))

    def build_set_op(self, arg: int) -> None:
        self.push(set(self.popn(arg)))

    def build_string_op(self, arg: int) -> None:
        pass

    def build_map_op(self, arg: int) -> None:
        values = self.popn(arg)
        keys = self.popn(arg)
        rez = dict()
        for item in range(arg):
            rez[keys[item]] = values[item]
        self.push(rez)

    def build_const_key_map_op(self, arg: int) -> None:
        num_values = int(arg)
        keys = self.pop()
        result_dict = dict.fromkeys(keys)
        values = self.popn(num_values)
        for i in range(num_values):
            result_dict[keys[i]] = values[i]
        self.push(result_dict)

    def build_tuple_unpack_op(self, arg: int) -> None:
        res = []
        for _ in range(arg):
            res += self.pop()
        self.push(tuple(res))

    def build_tuple_unpack_with_call_op(self, count: int) -> None:
        res = []
        for _ in range(count):
            res += self.pop()
        self.push(res)

    def build_set_unpack_op(self, count: int) -> None:
        res = []
        for _ in range(count):
            res += self.pop()
        self.push(set(res))

    def build_map_unpack_with_call_op(self, count: int) -> None:
        res = {}
        for _ in self.popn(count):
            res.update(self.pop())

        self.push(res)
        raise NotImplementedError

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-STORE_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2758
        """
        const = self.pop()
        self.locals[arg] = const

    def delete_name_op(self, arg: str) -> None:
        if arg in self.locals.keys():
            self.locals.pop(arg)
        elif arg in self.globals.keys():
            self.globals.pop(arg)
        elif arg in self.builtins.keys():
            self.builtins.pop(arg)
        else:
            assert False

    def unpack_sequence_op(self, count: str) -> None:
        seq = self.pop()
        for element in reversed(seq):
            self.push(element)

    def store_attr_op(self, arg: str) -> None:
        tos1, tos = self.topn(2)
        setattr(tos, arg, tos1)

    def delete_attr_op(self, namei: str) -> None:
        object_ = self.top()
        del object_

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def delete_global_op(self, arg: str) -> None:
        del self.globals[arg]

    def store_subscr_op(self, arg: str) -> None:
        tos2, tos1, tos = self.topn(3)
        tos1[tos] = tos2
        self.push(tos1[tos])

    def delete_subscr_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        del tos1[tos]

    def load_attr_op(self, arg: str) -> None:
        object_ = self.pop()
        value = getattr(object_, arg)
        self.push(value)

    def load_build_class_op(self, arg: str) -> None:
        self.push(builtins.__build_class__)

    def print_expr_op(self) -> None:
        print(self.pop())

    def load_fast_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        else:
            raise UnboundLocalError

    def delete_fast_op(self, arg: str) -> None:
        del self.locals[arg]

    def call_function_kw_op(self, arg: int) -> None:
        kwargs_key = self.pop()
        values = self.popn(arg)
        len_ = arg - len(kwargs_key)
        f = self.pop()
        kwargs = {}
        args = tuple(values[:len_])
        for index in range(len(kwargs_key)):
            kwargs[kwargs_key[index]] = values[index + len_]
        self.push(f(*args, **kwargs))

    def call_function_var_op(self, arg: str) -> None:
        arg = self.pop()
        f = self.pop()
        self.push(f(arg, {}))

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-MAKE_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4290

        Parse stack:
            https://github.com/python/cpython/blob/3.10/Objects/call.c#L612

        Call function in cpython:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4209
        """
        code = self.pop()

        # TODO: use arg to parse function defaults
        if arg & 8 == 8:
            self.pop()
        if arg & 2 == 2:
            self.pop()
        if arg & 1 == 1:
            self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # TODO: parse input arguments using code attributes such as co_argcount

            parsed_args: dict[str, tp.Any] = {}
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def build_slice_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        if arg == 2:
            self.push(slice(tos1, tos))
        elif arg == 3:
            self.push(slice(self.pop(), tos1, tos))

    def jump_absolute_op(self, target: int) -> None:
        self.current_offset = target - 2

    def jump_forward_op(self, delta: int) -> None:
        self.offset = delta

    def pop_jump_if_true_op(self, target: int) -> None:
        if self.pop():
            self.bytecode_counter = target

    def pop_jump_if_false_op(self, target: int) -> None:
        tos = self.top()
        if tos:
            self.bytecode_counter = target
        else:
            self.pop()

    def jump_if_not_exc_match_op(self, target: int) -> None:
        tos = self.top()
        tos1 = self.topn(-2)
        if not isinstance(tos1, tos):
            self.popn(2)
            self.bytecode_counter = target

    def jump(self, offset: int) -> None:
        self.current_offset = offset - 2

    def jump_if_true_or_pop_op(self, target: int) -> None:
        val = self.pop()
        if val:
            self.jump(target)

    def jump_if_false_or_pop_op(self, target: int) -> None:
        val = self.pop()
        if not val:
            self.jump(target)

    def for_iter_op(self, jump: int) -> None:
        tos = self.top()
        try:
            new_value = tos.__next__()
            self.push(new_value)
        except StopIteration:
            self.pop()
            self.jump_absolute_op(jump)

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-LOAD_GLOBAL

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2958
        """
        # TODO: parse all scopes
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def setup_loop_op(self, arg: int) -> None:
        self.loop_endings.append(arg)

    def pop_block_op(self, attr: str) -> None:
        self.block_stack.pop()

    def set_add_op(self, arg: int) -> None:
        add_value = self.pop()
        set.add(self.data_stack[-arg], add_value)

    def map_add_op(self, arg: int) -> None:
        key, value = self.popn(2)
        dict.__setitem__(self.data_stack[-arg], key, value)

    def list_append_op(self, arg: int) -> None:
        add_value = self.pop()
        list.append(self.data_stack[-arg], add_value)

    compare_operators = dict(zip(
        dis.cmp_op,
        [
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
            lambda a, b: a in b,
            lambda a, b: a not in b,
            lambda a, b: a is b,
            lambda a, b: a is not b,
            lambda a, b: issubclass(a, Exception) and issubclass(a, b),
        ]))

    def compare_op_op(self, op: str) -> None:
        first, second = self.popn(2)
        self.push(self.compare_operators[op](first, second))

    def binary_power_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 ** tos)

    def binary_multiply_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 * tos)

    def binary_true_divide_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 / tos)

    def binary_modulo_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 % tos)

    def binary_add_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 + tos)

    def binary_substact_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 - tos)

    def binary_subscr_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1[tos])

    def binary_and_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 & tos)

    def binary_xor_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 ^ tos)

    def binary_or_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 | tos)

    def binary_floor_divide_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 // tos)

    def binary_lshift_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 << tos)

    def binary_rshift_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 >> tos)

    def binary_subtract_op(self, arg: str) -> None:
        tos1, tos = self.popn(2)
        self.push(tos1 - tos)

    def binary_matrix_multiply_op(self, _: str) -> None:
        tos1 = self.pop()
        tos = self.pop()
        self.push(tos1 @ tos)

    def get_yield_from_iter_op(self, arg: str) -> None:
        if not (asyncio.iscoroutine(self.top())):
            self.get_iter_op(arg)

    def inplace_power_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] **= tos

    def inplace_multiply_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] *= tos

    def inplace_matrix_multiply_op(self, arg: str) -> None:
        tos = self.pop()
        tos1 = self.pop()
        tos1 @= tos
        self.push(tos1)

    def inplace_floor_divide_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] //= tos

    def inplace_true_divide_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] /= tos

    def inplace_modulo_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] %= tos

    def inplace_rshift_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] >>= tos

    def inplace_and_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] &= tos

    def inplace_xor_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] ^= tos

    def inplace_or_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] |= tos

    def inplace_add_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] += tos

    def inplace_subtract_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] -= tos

    def inplace_lshift_op(self, arg: str) -> None:
        tos = self.pop()
        self.data_stack[-1] <<= tos

    def load_assertion_error_op(self, arg: str) -> None:
        self.push(AssertionError(arg))

    def match_mapping_op(self) -> None:
        self.push(isinstance(self.top(), collections.abc.Mapping))

    def get_len_op(self, _: None) -> None:
        self.push(len(self.top()))

    def match_key_op(self) -> None:
        tos1, tos = self.topn(2)
        if set(tos).issubset(set(tos1)):
            self.push(tuple(tos))
            self.push(True)
        else:
            self.push(None)
            self.push(False)

    def setup_with_op(self, delta: int) -> None:
        top_ = self.top()
        self.push(top_.__exit__)
        top_a = top_.__enter__()
        self.push(top_a)

    def copy_dict_without_keys_op(self) -> None:
        tos1, tos = self.topn(2)
        self.data_stack[-1] = {key: value for key, value in tos1.items() if key not in tos}

    def list_extend_op(self, i: int) -> None:
        tos = self.pop()
        list.extend(self.data_stack[-i], tos)

    def set_update_op(self, i: int) -> None:
        tos = self.pop()
        set.update(self.data_stack[-i], tos)

    def dict_update_op(self, i: int) -> None:
        tos = self.pop()
        dict.update(self.data_stack[-i], tos)

    def dict_merge_op(self, i: int) -> None:
        tos = self.pop()
        if tos in dict(self.data_stack[-i]).keys():
            dict.update(self.data_stack[-i], tos)
        else:
            raise ValueError

    def format_value_op(self, arg: str) -> None:
        pass

    def import_name_op(self, arg: str) -> None:
        first, second = self.popn(2)
        self.push(__import__(arg, self.globals, self.locals, second, first))

    def import_from_op(self, arg: str) -> None:
        module = self.top()
        self.push(getattr(module, arg))

    def import_star_op(self, arg: str) -> None:
        module = self.pop()
        for attr in dir(module):
            if attr[0] != '_':
                self.locals[attr] = getattr(module, attr)

    def load_method_op(self, arg: str) -> None:
        object_ = self.pop()
        object_dict = object_.__class__.__dict__
        if arg in object_dict:
            self.push(object_)
            self.push(object_dict[arg])

    def raise_varargs_op(self, arg: str) -> None:
        if arg == 0:
            raise

    def list_to_tuple_op(self, arg: str) -> None:
        tuple_ = tuple(self.pop())
        self.push(tuple_)

    def contains_op_op(self, arg: str) -> None:
        left_operand, right_operand = self.popn(2)
        first_sit = left_operand not in right_operand
        second_sit = left_operand in right_operand
        if arg:
            self.push(first_sit)
        else:
            self.push(second_sit)

    def is_op_op(self, arg: str) -> None:
        left_operand, right_operand = self.popn(2)
        first_sit = left_operand is not right_operand
        second_sit = left_operand is right_operand
        if arg:
            self.push(first_sit)
        else:
            self.push(second_sit)

    def setup_finally_op(self, delta: int) -> None:
        pass

    def store_deref_op(self, arg: str) -> None:
        tos = self.pop()
        self.locals[arg] = tos

    def load_closure_op(self, i: str) -> None:
        self.push(i)

    def call_method_op(self, arg: int) -> None:
        arg = self.popn(arg)
        func = self.pop()
        object_ = self.pop()
        self.push(func(object_, *arg))

    def raise_varage_op(self, arg: str) -> None:
        if arg == 0:
            raise
        elif arg == 1:
            raise self.pop()
        elif arg == 2:
            tos1, tos = self.popn(2)
            raise tos1 from tos

    def extended_arg_op(self, arg: int) -> None:
        pass

    def setup_annotations_op(self, arg: str) -> None:
        if "__annotations__" not in self.locals:
            self.locals['__annotations__'] = {}

    def local_deref_op(self, i: str) -> None:
        self.push(self.locals[i])

    def pop_except_op(self, _: None) -> None:
        pass


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
