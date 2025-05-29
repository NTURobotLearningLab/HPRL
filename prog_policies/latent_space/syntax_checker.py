# Adapted from https://github.com/clvrai/leaps/blob/main/karel_env/tool/syntax_checker.py

import torch

from prog_policies.base import BaseDSL

STATE_MANDATORY_NEXT = 0
STATE_ACT_NEXT = 1
STATE_ACT_OR_CLOSE_NEXT = 2
STATE_CSTE_NEXT = 3
STATE_BOOL_NEXT = 4
STATE_POSTCOND_OPEN_PAREN = 5


class CheckerState(object):

    def __init__(self, state: int, next_mandatory: int, i_need_else_stack_pos: int, 
                 to_close_stack_pos: int, c_deep: int, next_actblock_open: int):
        self.state = state
        self.next_mandatory = next_mandatory
        self.i_need_else_stack_pos = i_need_else_stack_pos
        self.to_close_stack_pos = to_close_stack_pos
        self.c_deep = c_deep
        self.next_actblock_open = next_actblock_open
        self.i_need_else_stack = torch.tensor(128 * [False], dtype=torch.bool)
        self.to_close_stack = 128 * [None]

    def __copy__(self):
        new_state = CheckerState(self.state, self.next_mandatory,
                                 self.i_need_else_stack_pos, self.to_close_stack_pos,
                                 self.c_deep, self.next_actblock_open)
        for i in range(0, self.i_need_else_stack_pos+1):
            new_state.i_need_else_stack[i] = self.i_need_else_stack[i]
        for i in range(0, self.to_close_stack_pos+1):
            new_state.to_close_stack[i] = self.to_close_stack[i]
        return new_state

    def push_closeparen_to_stack(self, close_paren: int):
        self.to_close_stack_pos += 1
        self.to_close_stack[self.to_close_stack_pos] = close_paren

    def pop_close_paren(self) -> int:
        to_ret = self.to_close_stack[self.to_close_stack_pos]
        self.to_close_stack_pos -= 1
        return to_ret

    def paren_to_close(self):
        return self.to_close_stack[self.to_close_stack_pos]

    def make_next_mandatory(self, next_mandatory: int):
        self.state = STATE_MANDATORY_NEXT
        self.next_mandatory = next_mandatory

    def make_bool_next(self):
        self.state = STATE_BOOL_NEXT
        self.c_deep += 1

    def make_act_next(self):
        self.state = STATE_ACT_NEXT
        
    def make_act_or_close_next(self):
        self.state = STATE_ACT_OR_CLOSE_NEXT

    def close_cond_paren(self):
        self.c_deep -= 1
        if self.c_deep == 0:
            self.state = STATE_POSTCOND_OPEN_PAREN
        else:
            self.state = STATE_MANDATORY_NEXT
            # The mandatory next should already be "c)"

    def push_needelse_stack(self, need_else: bool):
        assert need_else == 0 or need_else == 1
        self.i_need_else_stack_pos += 1
        self.i_need_else_stack[self.i_need_else_stack_pos] = need_else

    def pop_needelse_stack(self) -> bool:
        to_ret = self.i_need_else_stack[self.i_need_else_stack_pos]
        self.i_need_else_stack_pos -= 1
        return to_ret

    def set_next_actblock(self, next_actblock: int):
        self.next_actblock_open = next_actblock

    def make_next_cste(self):
        self.state = STATE_CSTE_NEXT


class SyntaxVocabulary(object):

    def __init__(self, def_tkn, run_tkn,
                 m_open_tkn, m_close_tkn,
                 else_tkn, e_open_tkn,
                 c_open_tkn, c_close_tkn,
                 i_open_tkn, i_close_tkn,
                 while_tkn, w_open_tkn,
                 repeat_tkn, r_open_tkn,
                 not_tkn, pad_tkn):
        self.def_tkn = def_tkn
        self.run_tkn = run_tkn
        self.m_open_tkn = m_open_tkn
        self.m_close_tkn = m_close_tkn
        self.else_tkn = else_tkn
        self.e_open_tkn = e_open_tkn
        self.c_open_tkn = c_open_tkn
        self.c_close_tkn = c_close_tkn
        self.i_open_tkn = i_open_tkn
        self.i_close_tkn = i_close_tkn
        self.while_tkn = while_tkn
        self.w_open_tkn = w_open_tkn
        self.repeat_tkn = repeat_tkn
        self.r_open_tkn = r_open_tkn
        self.not_tkn = not_tkn
        self.pad_tkn = pad_tkn


class SyntaxChecker(object):

    def __init__(self, dsl: BaseDSL, device: torch.device, only_structure: bool = False):
        # check_type(args.no_cuda, bool)
        
        self.vocab = SyntaxVocabulary(dsl.t2i["DEF"], dsl.t2i["run"],
                                      dsl.t2i["m("], dsl.t2i["m)"], dsl.t2i["ELSE"], dsl.t2i["e("],
                                      dsl.t2i["c("], dsl.t2i["c)"], dsl.t2i["i("], dsl.t2i["i)"],
                                      dsl.t2i["WHILE"], dsl.t2i["w("], dsl.t2i["REPEAT"], dsl.t2i["r("],
                                      dsl.t2i["not"], dsl.t2i["<pad>"])

        self.device = device
        self.only_structure = only_structure
        
        open_paren_token = [tkn for tkn in dsl.get_tokens() if tkn.endswith("(")]
        close_paren_token = [tkn.replace("(",")") for tkn in open_paren_token]
        
        flow_leads = ["REPEAT", "WHILE", "IF", "IFELSE"]
        flow_leads = [flow_lead for flow_lead in flow_leads if flow_lead in dsl.get_tokens()]

        flow_need_bool = ["WHILE", "IF", "IFELSE"]
        flow_need_bool = [flow_need for flow_need in flow_need_bool if flow_need in dsl.get_tokens()]

        acts = dsl.actions
        bool_check = dsl.bool_features
        
        postcond_open_paren = ["i(", "w("]
        possible_mandatories = ["DEF", "run", "c)", "ELSE", "<pad>"] + open_paren_token
        
        self.open_parens = set([dsl.t2i[op] for op in open_paren_token])
        self.close_parens = set([dsl.t2i[op] for op in close_paren_token])
        self.if_statements = set([dsl.t2i[tkn] for tkn in ["IF", "IFELSE"]])
        self.op2cl = {}
        for op, cl in zip(open_paren_token, close_paren_token):
            self.op2cl[dsl.t2i[op]] = dsl.t2i[cl]
        self.need_else = {dsl.t2i["IF"]: False,
                          dsl.t2i["IFELSE"]: True}
        self.flow_lead = set([dsl.t2i[flow_lead_tkn] for flow_lead_tkn in flow_leads])
        
        if self.only_structure:
            self.effect_acts = set()
            self.range_cste = set()
            self.bool_checks = set()
        else:
            self.effect_acts = set([dsl.t2i[act_tkn] for act_tkn in acts])
            self.range_cste = set([idx for tkn, idx in dsl.t2i.items() if tkn.startswith("R=")])
            self.bool_checks = set([dsl.t2i[bcheck] for bcheck in bool_check])
        if "<HOLE>" in dsl.t2i.keys():
            self.effect_acts.add(dsl.t2i["<HOLE>"])
            self.bool_checks.add(dsl.t2i["<HOLE>"])
            self.range_cste.add(dsl.t2i["<HOLE>"])

        self.act_acceptable = self.effect_acts | self.flow_lead
        self.flow_needs_bool = set([dsl.t2i[flow_tkn] for flow_tkn in flow_need_bool])
        self.postcond_open_paren = set([dsl.t2i[op] for op in postcond_open_paren])

        tt = torch.cuda if 'cuda' in self.device.type else torch
        self.vocab_size = len(dsl.t2i)
        self.mandatories_mask = {}
        for mand_tkn in possible_mandatories:
            mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,dsl.t2i[mand_tkn]] = 0
            self.mandatories_mask[dsl.t2i[mand_tkn]] = mask
        self.act_next_mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
        for act_tkn in self.act_acceptable:
            self.act_next_mask[0,0,act_tkn] = 0
        self.act_or_close_next_masks = {}
        for close_tkn in self.close_parens:
            mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,close_tkn] = 0
            for effect_idx in self.effect_acts:
                mask[0,0,effect_idx] = 0
            for flowlead_idx in self.flow_lead:
                mask[0,0,flowlead_idx] = 0
            self.act_or_close_next_masks[close_tkn] = mask
        self.range_mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
        for ridx in self.range_cste:
            self.range_mask[0,0,ridx] = 0
        self.boolnext_mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
        for bcheck_idx in self.bool_checks:
            self.boolnext_mask[0,0,bcheck_idx] = 0
        self.boolnext_mask[0,0,self.vocab.not_tkn] = 0
        self.postcond_open_paren_masks = {}
        for tkn in self.postcond_open_paren:
            mask = tt.BoolTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,tkn] = 0
            self.postcond_open_paren_masks[tkn] = mask

    def forward(self, state: CheckerState, new_idx: int):
        # Whatever happens, if we open a paren, it needs to be closed
        if new_idx in self.open_parens:
            state.push_closeparen_to_stack(self.op2cl[new_idx])
        if new_idx in self.close_parens:
            paren_to_end = state.pop_close_paren()
            assert(new_idx == paren_to_end)

        if state.state == STATE_MANDATORY_NEXT:
            assert(new_idx == state.next_mandatory)
            if new_idx == self.vocab.def_tkn:
                state.make_next_mandatory(self.vocab.run_tkn)
            elif new_idx == self.vocab.run_tkn:
                state.make_next_mandatory(self.vocab.m_open_tkn)
            elif new_idx == self.vocab.else_tkn:
                state.make_next_mandatory(self.vocab.e_open_tkn)
            elif new_idx in self.open_parens:
                if new_idx == self.vocab.c_open_tkn:
                    state.make_bool_next()
                else:
                    state.make_act_next()
            elif new_idx == self.vocab.c_close_tkn:
                state.close_cond_paren()
            elif new_idx == self.vocab.pad_tkn:
                # Should this be at the top?
                # Keep the state in mandatory next, targetting <pad>
                # Once you go <pad>, you never go back.
                pass
            else:
                raise NotImplementedError

        elif state.state == STATE_ACT_NEXT:
            assert(new_idx in self.act_acceptable)

            if new_idx in self.flow_needs_bool:
                state.make_next_mandatory(self.vocab.c_open_tkn)
                # If we open one of the IF statements, we need to keep track if
                # it's one with a else statement or not
                if new_idx in self.if_statements:
                    state.push_needelse_stack(self.need_else[new_idx])
                    state.set_next_actblock(self.vocab.i_open_tkn)
                elif new_idx == self.vocab.while_tkn:
                    state.set_next_actblock(self.vocab.w_open_tkn)
                else:
                    raise NotImplementedError
            elif new_idx == self.vocab.repeat_tkn:
                state.make_next_cste()
            elif new_idx in self.effect_acts:
                state.make_act_or_close_next()
                
        elif state.state == STATE_ACT_OR_CLOSE_NEXT:
            assert(new_idx in self.act_acceptable | self.close_parens)

            if new_idx in self.flow_needs_bool:
                state.make_next_mandatory(self.vocab.c_open_tkn)
                # If we open one of the IF statements, we need to keep track if
                # it's one with a else statement or not
                if new_idx in self.if_statements:
                    state.push_needelse_stack(self.need_else[new_idx])
                    state.set_next_actblock(self.vocab.i_open_tkn)
                elif new_idx == self.vocab.while_tkn:
                    state.set_next_actblock(self.vocab.w_open_tkn)
                else:
                    raise NotImplementedError
            elif new_idx == self.vocab.repeat_tkn:
                state.make_next_cste()
            elif new_idx in self.effect_acts:
                pass
            elif new_idx in self.close_parens:
                if new_idx == self.vocab.i_close_tkn:
                    need_else = state.pop_needelse_stack()
                    if need_else:
                        state.make_next_mandatory(self.vocab.else_tkn)
                    else:
                        state.make_act_or_close_next()
                elif new_idx == self.vocab.m_close_tkn:
                    state.make_next_mandatory(self.vocab.pad_tkn)
                else:
                    state.make_act_or_close_next()
            else:
                raise NotImplementedError

        elif state.state == STATE_CSTE_NEXT:
            assert(new_idx in self.range_cste)
            state.make_next_mandatory(self.vocab.r_open_tkn)

        elif state.state == STATE_BOOL_NEXT:
            if new_idx in self.bool_checks:
                state.make_next_mandatory(self.vocab.c_close_tkn)
            elif new_idx == self.vocab.not_tkn:
                state.make_next_mandatory(self.vocab.c_open_tkn)
            else:
                raise NotImplementedError

        elif state.state == STATE_POSTCOND_OPEN_PAREN:
            assert(new_idx in self.postcond_open_paren)
            assert(new_idx == state.next_actblock_open)
            state.make_act_next()

        else:
            raise NotImplementedError

    def allowed_tokens(self, state: CheckerState):
        if state.state == STATE_MANDATORY_NEXT:
            # Only one possible token follows
            return self.mandatories_mask[state.next_mandatory]
        elif state.state == STATE_ACT_NEXT:
            return self.act_next_mask
        elif state.state == STATE_ACT_OR_CLOSE_NEXT:
            # Either an action, a control flow statement or a closing of an open-paren
            return self.act_or_close_next_masks[state.paren_to_close()]
        elif state.state == STATE_CSTE_NEXT:
            return self.range_mask
        elif state.state == STATE_BOOL_NEXT:
            return self.boolnext_mask
        elif state.state == STATE_POSTCOND_OPEN_PAREN:
            return self.postcond_open_paren_masks[state.next_actblock_open]

    def get_sequence_mask(self, state: CheckerState, inp_sequence: list):
        if len(inp_sequence) == 1:
            self.forward(state, inp_sequence[0])
            return self.allowed_tokens(state).squeeze(0)
        else:
            tt = torch.cuda if 'cuda' in self.device.type else torch
            mask_infeasible_list = []
            mask_infeasible = tt.BoolTensor(1, len(inp_sequence), self.vocab_size)
            for stp_idx, inp in enumerate(inp_sequence):
                self.forward(state, inp)
                mask_infeasible_list.append(self.allowed_tokens(state))
            torch.cat(mask_infeasible_list, 1, out=mask_infeasible)
            return mask_infeasible

    def get_initial_checker_state(self):
        return CheckerState(STATE_MANDATORY_NEXT, self.vocab.def_tkn,
                            -1, -1, 0, -1)

    def get_initial_checker_state2(self):
        return CheckerState(STATE_MANDATORY_NEXT, self.vocab.m_open_tkn,
                            -1, -1, 0, -1)
