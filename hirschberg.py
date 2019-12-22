class Hirschberg:

    def __init__(self, v: str, w: str, scoring_map: map = None):
        # scoring_map = {"match": int_match_score, "mismatch": int_mismatch_score, "gap": int_gap_score}
        if not scoring_map:
            raise Exception("No Scoring function (A python Dictionary) provided!")
        self.v = "0" + v  # Input string that is on the row of the DP matrix (with i as its pointer for index [row])
        self.w = "0" + w  # Input string that is on the column of the DP matrix (with j as its pointer for index [column])
        self.match_score = scoring_map["match"]  # Scoring map. Needs to be provided by user
        self.mismatch_score = scoring_map["mismatch"]  # Scoring map. Needs to be provided by user
        self.gap_score = scoring_map["gap"]  # Scoring map. Needs to be provided by user
        self.alignment_tuples_list = []  # Alignment tuples calculated by the hirschberg algorithm
        self.final_alignment_tuples_list = []  # The final list of tuples for the maximum weight global alignment for two strings
        self.final_alignment_strings_list = []  # The final alignment presented in strings for easy visualization
        self.column_dp = [[],
                          []]  # The space efficient 2 columns used for hirschberg DP recurrence with scoring function
        self.backtrace_steps_map = {}  ## Key is the step number: value is probably tuples to hirschberg functions etc to know each step process for visualization

    @staticmethod
    def get_middle_j(j: int, j_plum: int) -> int:
        return j + (j_plum - j) // 2

    @staticmethod
    def reversed_string_for_suffix(string: str) -> str:
        return string[::-1]

    def theta_score(self, char1: str, char2: str, option="matching") -> int:
        if option == "matching":
            return self.match_score if char1 == char2 else self.mismatch_score
        else: # its a gap action (insertion or deletion)
            return self.gap_score

    def report(self, i_optimal: int, j_optimal: int, point_weight=None) -> None:
        if point_weight:
            self.alignment_tuples_list.append((i_optimal, j_optimal, point_weight))
        else:
            self.alignment_tuples_list.append((i_optimal, j_optimal))

    def argmax_weight(self, row_index_start: int, row_index_end: int, column_index_start: int, column_index_end: int,
                      middle_j: int) -> (int, int):
        print("(i,j,i',j') := ", (row_index_start, column_index_start, row_index_end, column_index_end))
        prefix_i_score_column = self.run_column_dp(row_index_start, row_index_end, column_index_start, column_index_end, middle_j)
        suffix_i_score_column = self.run_column_dp(row_index_start, row_index_end, column_index_start, column_index_end, middle_j,
                                                   prefix_option=False)  # For suffix, need to reverse/flip sub matrix
        arg_weight_list = [prefix_score + suffix_score for prefix_score, suffix_score in zip(prefix_i_score_column,
                                                                                             suffix_i_score_column)]

        print("      prefix_i_column_score: ", prefix_i_score_column)
        print("      suffix_i_column_score: ", suffix_i_score_column)
        print("              Final_arg_weight_score: ", arg_weight_list)
        print("=========================================================")

        # Need to link the current index of the argmax/arg_weight_list to the actual row i index of the string v
        target_i_start = 0
        arg_max_weight_score = float('-inf')
        for index, weight_score in enumerate(arg_weight_list):
            if weight_score > arg_max_weight_score:
                target_i_start = index
                arg_max_weight_score = weight_score

        # Need to link the current index of the argmax/arg_weight_list to the actual row i index of the string v
        if row_index_start == 0:
            return target_i_start, arg_max_weight_score
        else:
            return (target_i_start + row_index_start - 1), arg_max_weight_score  # # THIS <=====!!!!

    def dp_weight_score_arg_max(self, prev_dp_column: list, cur_dp_column: list,
                                cur_row_position: int,
                                v_char: str, w_char: str) -> int:

        insertion_score = cur_dp_column[cur_row_position - 1] + self.theta_score(v_char, w_char, "insertion")
        deletion_score = prev_dp_column[cur_row_position] + self.theta_score(v_char, w_char, "deletion")
        matching_score = prev_dp_column[cur_row_position - 1] + self.theta_score(v_char, w_char)
        return max(insertion_score, deletion_score, matching_score)

    def run_column_dp(self, row_index_start: int, row_index_end: int, column_index_start: int, column_index_end: int,
                      middle_j: int, prefix_option: bool = True):
        # Extract the substrings that are now used for both v and w in the current hirschberg run
        # initialize the dp first column (The column that is 0)
        print("     Middle j value :=  ", middle_j)
        # v_slice = self.v[row_index_start:row_index_end + 1]
        # w_slice = self.w[column_index_start:column_index_end + 1]
        if prefix_option:
            v_slice = self.v[row_index_start:row_index_end + 1]
            w_slice = self.w[column_index_start:middle_j + 1]
            if len(v_slice) > 0 and v_slice[0] == "0":
                v_plum = v_slice
            else:
                v_plum = "0" + v_slice

            if len(w_slice) > 0 and w_slice[0] == "0":
                w_plum = w_slice
            else:
                w_plum = '0' + w_slice
            print("    Running Column DP for Prefix: ")
            print("      v_plum: ", v_plum)
            print("      w_plum: ", w_plum)
            # dp_update_j_end = middle_j - row_index_start + 1
        else:  # Preparing data for calculating suffix
            v_slice = self.v[row_index_start:row_index_end + 1]
            w_slice = self.w[middle_j + 1:column_index_end + 1]
            if len(v_slice) > 0 and v_slice[0] == "0":
                v_slice = v_slice[1:]
            if len(w_slice) > 0 and w_slice[0] == "0":
                w_slice = w_slice[1:]
            v_plum = '0' + Hirschberg.reversed_string_for_suffix(v_slice)
            w_plum = '0' + Hirschberg.reversed_string_for_suffix(w_slice)
            print("    Running Column DP for Suffix: ")
            print("      v_plum: ", v_plum)
            print("      w_plum: ", w_plum)
            # dp_update_j_end = column_index_end - (middle_j + 1) + 1

        # Initialize the first column of the 2 columns of the linear-space DP (index 0 for w_plum)
        temp_dp = [[0], []]
        for i in range(1, len(v_plum)):
            dp_value = temp_dp[0][i - 1] + self.theta_score(v_plum[i], w_plum[0], "initialization")
            temp_dp[0].append(dp_value)

        # Recurrence for each column j of w_plum. Start for index 1 as index 0 was initialized previously
        prev_dp_column = temp_dp[0]  # [0, -1n, -2n, -3n, ....]
        cur_dp_column = temp_dp[1]  # []
        # column j needs to be iterated to the middle_j column for prefix from the column start and
        # middle_j + 1 column for suffix from the column end
        for j in range(1, len(w_plum)):
            print("        Calculating DP column weight points - Previous column: ", prev_dp_column)
            print("        Calculating DP column weight points - Current column: ", cur_dp_column)
            w_plum_j_char = w_plum[j]
            # Go through each row
            for i in range(len(v_plum)):
                v_plum_i_char = v_plum[i]
                # The very first row that matches to the character '0' for the base case of the DP recurrence
                # This would only be updated from the left column value
                if i == 0:
                    cur_weight_score = prev_dp_column[i] + self.gap_score
                    cur_dp_column.append(cur_weight_score)  # Look at the left column score for row 1 (character '0')
                else:
                    arg_max_weight_score = self.dp_weight_score_arg_max(prev_dp_column, cur_dp_column, i,
                                                                    v_plum_i_char, w_plum_j_char)
                    cur_dp_column.append(arg_max_weight_score)
            # Each column means a new/current dp_column to calculate. Update the current one to become the next iteration's
            # previous dp column.
            if j != len(w_plum) - 1:
                prev_dp_column = list(cur_dp_column)
                cur_dp_column = []

        temp_dp[0] = list(prev_dp_column)
        temp_dp[1] = list(cur_dp_column) if prefix_option else list(cur_dp_column)[::-1]
        print("                FINAL Calculating DP column weight points: ", temp_dp)

        # Once we calculated that last column of the DP weight. It is either the column of scores for prefix or suffix
        # in order to calculate weight(i) argmax_weight. Need to reverse the column score if its calculating suffix
        # return temp_dp[1] if prefix_option else temp_dp[1][::-1]
        return temp_dp[1]

    def hirschberg(self, i: int, j: int, i_plum: int, j_plum: int):
        if j_plum - j > 1:
            middle_j = Hirschberg.get_middle_j(j, j_plum)
            i_star, point_weight = self.argmax_weight(i, i_plum, j, j_plum, middle_j)
            ## self.report(i_star, middle_j, point_weight)
            self.report(i_star, middle_j)
            self.hirschberg(i, j, i_star, middle_j)
            self.hirschberg(i_star, middle_j, i_plum, j_plum)
        else:  # Reporting the leave values of the Hirschberg Recursion tree
            self.report(i, j)
            self.report(i_plum, j_plum)
            ## Extra Changes could be made here for more detailed examples of weights and even use for futher
            ## visualization enhancements
            # self.report(i, j, '-')
            # self.report(i_plum, j_plum, '-')

    def post_process_construct_hirschberg_alignment(self, alignment_tuples_list: list) -> list:
        # Four different ways between points
        # Every tuple entry corresponds to a point in a single column. Go through all column points from the very end
        # to the start, if both points are not either linked exactly side by side (horizontally) nor diagonally. Then
        # Find all possible paths from the later point to its earlier point, calculate the weights to find the possible
        # path (extra tuple points) that would generate the hirschberg weight
        print("::::::: Post processing constructing hirschberg alignment by going through every generated hirschberg coordinate ::::::::")
        temp_final_alignment_tuple_list = []
        for i in range(len(alignment_tuples_list) - 1, 0, -1):
            # cur_tuple = alignment_tuples_list[i]
            # prev_tuple = alignment_tuples_list[i - 1]
            # Initialize the third tuple element to hold weight for backtracking
            cur_tuple = tuple(list(alignment_tuples_list[i]) + [0])
            prev_tuple = tuple(list(alignment_tuples_list[i - 1]) + [0])
            print(cur_tuple, prev_tuple)
            reconstructed_upward_path_list = self.reconstruct_alignment(cur_tuple, prev_tuple)
            temp_final_alignment_tuple_list.append(cur_tuple)
            for recon_tuple in reconstructed_upward_path_list:
                if type(recon_tuple) == list:
                    # If there are paths that are derived between two hirschberg coordinates, then it would be a list of tuples
                    for reconstructed_path_tuple in recon_tuple:
                        temp_final_alignment_tuple_list.append(reconstructed_path_tuple)
                else:
                    temp_final_alignment_tuple_list.append(recon_tuple)

        # Add in the final (0,0,0) tuple (Initial coordinate)
        temp_final_alignment_tuple_list.append((0, 0, 0))
        temp_final_alignment_tuple_list.reverse()

        self.final_alignment_tuples_list = temp_final_alignment_tuple_list
        print("    Final Alignment Tuple List (i coordinate, j coordinate, [weight values for future use]) : ",
              self.final_alignment_tuples_list)
        return self.get_final_alignment_tuples_list()

    def reconstruct_alignment(self, current_tuple: tuple, prev_tuple: tuple) -> list:
        # Only returns the missing tuples from one tuple to its previous tuple calculated by the hirschberg algorithm
        # Both starting and ending tuples are not included
        def dfs(cur_tuple, target_tuple, diag_shift, left_shift, result_list, temp_list):
            # Base case. When we reached the target_tuple points from cur_tuple points
            if cur_tuple[0] == target_tuple[0] and cur_tuple[1] == target_tuple[1]:
                if cur_tuple[2] == target_tuple[2]:
                    result_list.append(list(temp_list))
                else:
                    # temp_list.pop()
                    pass
                return

            if diag_shift | left_shift != 0:  # Only with the option of going up instead
                updated_cur_tuple = (cur_tuple[0] - 1, cur_tuple[1], cur_tuple[2] + self.gap_score)
                next_temp_list = temp_list + [updated_cur_tuple]
                dfs(updated_cur_tuple, target_tuple,
                    diag_shift, left_shift, result_list, next_temp_list)
            else:  # Still haven't moved to the left column. Choose to go diag or left
                if cur_tuple[0] == target_tuple[0]:  # When both are on the same row but different column. Can only move left
                    updated_cur_tuple = (cur_tuple[0], cur_tuple[1] - 1, cur_tuple[2] + self.gap_score)
                    dfs(updated_cur_tuple, target_tuple,
                        diag_shift, 1, result_list, temp_list + [updated_cur_tuple])
                else:  # We are able to choose any. Either go up, to the left or diagonally
                    # Choose to move diagonally
                    updated_cur_tuple = (cur_tuple[0] - 1, cur_tuple[1] - 1,
                                         cur_tuple[2] + self.theta_score(self.v[cur_tuple[0]], self.w[cur_tuple[1]]))
                    temp_list.append(updated_cur_tuple)
                    dfs(updated_cur_tuple, target_tuple,
                        1, left_shift, result_list, temp_list)
                    try:
                        # Backtrace
                        temp_list.pop()
                    except Exception as e:
                        print(e)
                        print("     :::::      :::::: ")
                        print(updated_cur_tuple)

                    # Choose to move up
                    updated_cur_tuple = (cur_tuple[0] - 1, cur_tuple[1], cur_tuple[2] + self.gap_score)
                    temp_list.append(updated_cur_tuple)
                    dfs(updated_cur_tuple, target_tuple,
                        diag_shift, left_shift, result_list, temp_list)
                    # Backtrace
                    temp_list.pop()
                    # Choose to move left
                    updated_cur_tuple = (cur_tuple[0], cur_tuple[1] - 1, cur_tuple[2] + self.gap_score)
                    temp_list.append(updated_cur_tuple)
                    dfs(updated_cur_tuple, target_tuple,
                        diag_shift, 1, result_list, temp_list)

        intermediary_vertices_list = []
        dfs(current_tuple, prev_tuple, 0, 0, intermediary_vertices_list, [])
        return intermediary_vertices_list

    def get_final_alignment_tuples_list(self):
        return self.final_alignment_tuples_list

    def generate_final_alignment_strings_list(self):
        #  The extra (0,0) tuple  won't be in the final alignment tuple list.
        # Every tuple should exactly match with the v & w strings index
        final_alignment_tuple = self.get_final_alignment_tuples_list()[::-1]  # reverse it for backtrace
        v_alignment_string = ""  # would need to reverse later
        w_alignment_string = ""  # would need to reverse for correct alignment
        print("============ Final alignment tuple ==========")

        print(final_alignment_tuple)
        for index in range(len(final_alignment_tuple)):
            cur_tuple = final_alignment_tuple[index]
            prev_tuple = final_alignment_tuple[index-1]
            # == Ignore duplicate tuples ==
            if cur_tuple == prev_tuple:
                continue

            if cur_tuple[0] == prev_tuple[0]:  # On the same row, meaning only moved to the previous column horizontally
                v_alignment_string += "-"
                w_alignment_string += self.w[cur_tuple[1] + 1]
            elif cur_tuple[0] == prev_tuple[0] - 1 and cur_tuple[1] == prev_tuple[1] - 1:  # Went back diagonally
                v_alignment_string += self.v[cur_tuple[0] + 1]
                w_alignment_string += self.w[cur_tuple[1] + 1]
            elif cur_tuple[0] == prev_tuple[0] - 1:  # Went upwards
                v_alignment_string += self.v[cur_tuple[0] + 1]
                w_alignment_string += "-"
            else:
                pass

        # Reverse alignments back
        v_alignment_string = v_alignment_string[::-1]
        w_alignment_string = w_alignment_string[::-1]
        self.final_alignment_strings_list = [v_alignment_string, w_alignment_string]
        return self.final_alignment_strings_list

    def run(self):
        print("======== Hirschberg Running =========\n")
        self.hirschberg(0, 0, len(self.v) - 1, len(self.w) - 1)
        print("======== Hirschberg Finished! =========\n")
        # Keep only unique tuple sets of the DP coordinates
        unique_alignment_tuple_set = set(self.alignment_tuples_list)
        self.alignment_tuples_list = list(unique_alignment_tuple_set)

        self.alignment_tuples_list.sort(key=lambda tuple_entry: tuple_entry[1])  # sort out the final tuples based on column j index
        print("\n++++++++  Final Alignment Tuple List of (i,j) for each aligned character pair in the sequences identified by each Hirschberg run ++++++++")
        print("+++++++>  ", self.alignment_tuples_list, "\n")  # Tuples of (i,j) for each aligned character pair in the sequences
        ## Post process to find the whole entire sequence alignment
        self.post_process_construct_hirschberg_alignment(self.alignment_tuples_list)
        final_alignment_list = self.generate_final_alignment_strings_list()
        print("\n\n ------- Original Inputs --------")
        print("     w: ", self.w[1:])
        print("     v: ", self.v[1:])
        print("\n\n***** Final Alignment *****")
        print("     W: ", final_alignment_list[1])
        print("     V: ", final_alignment_list[0])


if __name__ == "__main__":
    scoring_map = {
        "match": 1,
        "mismatch": -1,
        "gap": -1
    }

    # v = "ATGTC"
    # w = "ATCGC"
    v = "CT"
    w = "GCAT"

    Hirschberg_object = Hirschberg(v, w, scoring_map)
    Hirschberg_object.run()

    # middle_j = Hirschberg.get_middle_j(0, len(w))
    # print(middle_j)
    # Hirschberg_object.run_column_dp(0, len(v), 0, len(w))
    # print(Hirschberg_object.argmax_weight(0, len(v), 0, len(w)))
