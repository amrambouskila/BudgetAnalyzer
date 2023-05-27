import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
import seaborn as sns
from enum import Enum
from datetime import datetime
from sklearn.linear_model import LinearRegression
from logger import create_logger
import mplcursors
import matplotlib.patches as mpatches


matplotlib.use('Qt5Agg', force=True)
sns.set()


class Budget:
    def __init__(self, monthly_income: float, monthly_expenses: dict, monthly_diet: dict, yearly_expenses: dict, investment_portfolio: dict, available_funds: float = 0.):
        self.monthly_income = monthly_income
        self.monthly_expenses = monthly_expenses
        self.monthly_diet = monthly_diet
        self.yearly_expenses = yearly_expenses
        self.available_funds = monthly_income + available_funds

        self.monthly_expenses_fund = 0
        self.calculate_monthly_expense_funds()

        self.categorized_monthly_expenses = {}
        for expense in monthly_expenses.keys():
            if monthly_expenses[expense]['category'] in self.categorized_monthly_expenses.keys():
                self.categorized_monthly_expenses[monthly_expenses[expense]['category']] += monthly_expenses[expense]['amount']
            else:
                self.categorized_monthly_expenses[monthly_expenses[expense]['category']] = monthly_expenses[expense]['amount']

        for expense in yearly_expenses.keys():
            if yearly_expenses[expense]['category'] in self.categorized_monthly_expenses.keys():
                self.categorized_monthly_expenses[yearly_expenses[expense]['category']] += yearly_expenses[expense]['amount'] / 12
            else:
                self.categorized_monthly_expenses[yearly_expenses[expense]['category']] = yearly_expenses[expense]['amount'] / 12

        self.yearly_expenses_fund = 0
        self.calculate_yearly_expense_funds()

        self.monthly_diet_fund = 0
        self.calculate_monthly_diet_funds()

        self.investment_fund = 0
        self.calculate_investment_funds()
        self.fund_distribution = pd.DataFrame({'Category': ['Income',
                                                            'Monthly Expenses',
                                                            'Monthly Diet',
                                                            'Yearly Expenses',
                                                            'Investments'],
                                               'Budget Amount': [self.monthly_income,
                                                                 -self.monthly_expenses_fund,
                                                                 -self.monthly_diet_fund,
                                                                 -self.yearly_expenses_fund,
                                                                 -self.investment_fund]
                                               })

        self._investment_portfolio = {}
        self.investment_portfolio = investment_portfolio
        self.print_budget()

    @property
    def investment_portfolio(self):
        return self._investment_portfolio

    @investment_portfolio.setter
    def investment_portfolio(self, investment_portfolio):
        for investment in investment_portfolio.keys():
            if investment in ['BTC', 'ETH']:
                self._investment_portfolio[investment] = {'Monthly': investment_portfolio[investment] * self.investment_fund, 'Daily': investment_portfolio[investment] * self.investment_fund / 30}
            else:
                self._investment_portfolio[investment] = {'Monthly': investment_portfolio[investment] * self.investment_fund, 'Daily': investment_portfolio[investment] * self.investment_fund / 20}

    def calculate_yearly_expense_funds(self):
        for expense in self.yearly_expenses.keys():
            expense_amount = self.yearly_expenses[expense]['amount'] / 12
            self.yearly_expenses_fund += expense_amount
            self.available_funds -= expense_amount

    def calculate_monthly_expense_funds(self):
        for expense in self.monthly_expenses.keys():
            expense_amount = self.monthly_expenses[expense]['amount']
            self.monthly_expenses_fund += expense_amount
            self.available_funds -= expense_amount

    def calculate_monthly_diet_funds(self):
        monthly_diet = sum(self.monthly_diet[food] for food in self.monthly_diet.keys())
        self.monthly_diet_fund += monthly_diet
        self.available_funds -= monthly_diet

    def calculate_investment_funds(self):
        monthly_investment = self.available_funds / 2
        self.available_funds -= monthly_investment
        self.investment_fund += monthly_investment

    def print_budget(self):
        print(f'Monthly Income: {self.monthly_income}')
        print(f'Monthly Expense Funds: {self.monthly_expenses_fund}')
        print(f'Monthly Diet Funds: {self.monthly_diet_fund}')
        print(f'Yearly Expense Funds: {self.yearly_expenses_fund}')
        print(f'Investment Funds: {self.investment_fund}')
        print(f'Investment Portfolio: {self.investment_portfolio}')
        print(f'Available Funds: {self.available_funds}')
        print(f'Total Daily Investment: {sum([self.investment_portfolio[investment]["Daily"] for investment in self.investment_portfolio])}')


class BudgetAnalyzer:
    months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'June', 7: 'July',
              8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    def __init__(self, budget: Budget, start_date: str, actual_spending: str, periods: int = 12):
        self.start_date = start_date
        self.budget = budget
        self.actual_spending = pd.read_csv(actual_spending)
        self.periods = periods
        self.comparison = None
        self.model = None
        self.visualizations = {}

        self.actual_spending = self.actual_spending[self.actual_spending['Account Number'] == 5525]
        self.actual_spending.drop(['Original Date', 'Account Type', 'Account Name', 'Account Number', 'Institution Name', 'Custom Name', 'Note', 'Ignored From', 'Tax Deductible'], axis=1, inplace=True)
        self.actual_spending['Date'] = pd.to_datetime(self.actual_spending['Date'])
        self.actual_spending = self.actual_spending[self.actual_spending['Date'] >= pd.Timestamp(self.start_date)].reset_index()
        for transaction in range(self.actual_spending.shape[0]):
            if self.actual_spending.loc[transaction, 'Category'] in ['Auto & Transport', 'Credit Card Payment']:
                self.actual_spending.loc[transaction, 'Category'] = 'Transportation'
            if self.actual_spending.loc[transaction, 'Category'] == 'Bills & Utilities':
                self.actual_spending.loc[transaction, 'Category'] = 'Utilities'
            if self.actual_spending.loc[transaction, 'Category'] in ['Dining & Drinks', 'Entertainment & Rec.', 'Shopping']:
                self.actual_spending.loc[transaction, 'Category'] = 'Entertainment'
            if self.actual_spending.loc[transaction, 'Category'] in ['Groceries', 'Health & Wellness']:
                self.actual_spending.loc[transaction, 'Category'] = 'Health'
            if self.actual_spending.loc[transaction, 'Category'] == 'Family Care':
                self.actual_spending.loc[transaction, 'Category'] = 'Donation'
            if self.actual_spending.loc[transaction, 'Category'] in ['Fees', 'Uncategorized']:
                self.actual_spending.loc[transaction, 'Category'] = 'Miscellaneous'
            if self.actual_spending.loc[transaction, 'Category'] == 'Home & Garden':
                self.actual_spending.loc[transaction, 'Category'] = 'Shelter'
            if self.actual_spending.loc[transaction, 'Category'] == 'Software & Tech':
                self.actual_spending.loc[transaction, 'Category'] = 'Education'

        self.actual_monthly_spending = self.actual_spending.groupby([self.actual_spending['Date'].dt.month])['Amount'].sum().reset_index()
        self.actual_monthly_categorical_spending = self.actual_spending.groupby([self.actual_spending['Date'].dt.month, 'Category'])['Amount'].sum().reset_index()
        self.average_monthly_spending = self.actual_monthly_spending['Amount'].mean()
        self.average_monthly_categorical_spending = self.actual_monthly_categorical_spending.groupby('Category')['Amount'].mean()

        for cat in [*self.actual_monthly_categorical_spending['Category'].unique()]:
            if cat not in self.budget.categorized_monthly_expenses.keys() and cat != 'Income':
                self.budget.categorized_monthly_expenses[cat] = 0

        self.visualize_budget()
        for month in list(set(self.actual_spending['Date'].dt.month)):
            self.visualize_budget(month=month)
        self.visualize_budget()
        print('Visualizations Complete!')

    @staticmethod
    def adjust_legend(legend_lines, ax, colors):
        legend = ax.legend(handles=legend_lines, loc='best', fontsize=5, handleheight=0.5, ncol=3)
        legend.set_draggable(True)

        # Adjust legend properties
        legend.get_frame().set_alpha(0.7)  # Set legend transparency
        legend.get_frame().set_facecolor('white')  # Set legend background color
        legend.get_frame().set_linewidth(0.5)  # Set legend border width
        legend.get_frame().set_edgecolor('black')  # Set legend border color
        proxy_artists = [mpatches.Rectangle((0, 0), 10, 10, color=color) for color in colors]

        # Set the custom proxy artists for legend handles
        legend.legendHandles = proxy_artists

        for label in legend.get_texts():
            label.set_fontsize(8)

    def visualize_budget(self, month=None, average=False):
        fig = plt.figure(figsize=(12, 10))
        fig.canvas.mpl_connect('motion_notify_event', self.mouse_motion_handler)

        if month:
            ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=2)
            # self.visualize_fund_distribution(ax=ax1, month=month)

            ax2 = plt.subplot2grid((5, 4), (2, 0), colspan=2, rowspan=1)
            # self.visualize_investment_distribution(ax=ax2, month=month)

            ax3 = plt.subplot2grid((5, 4), (2, 2), colspan=2, rowspan=1)
            # self.visualize_monthly_diet_distribution(ax=ax3, month=month)

            ax4 = plt.subplot2grid((5, 4), (3, 0), colspan=2, rowspan=1)
            # self.visualize_yearly_expense_distribution(ax=ax4, month=month)

            ax5 = plt.subplot2grid((5, 4), (3, 2), colspan=2, rowspan=1)
            self.visualize_monthly_expense_distribution(ax=ax5, month=month)
        elif average:
            ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=2)
            self.visualize_fund_distribution(ax=ax1, average=average)

            ax2 = plt.subplot2grid((5, 4), (2, 0), colspan=2, rowspan=1)
            self.visualize_investment_distribution(ax=ax2, average=average)

            ax3 = plt.subplot2grid((5, 4), (2, 2), colspan=2, rowspan=1)
            self.visualize_monthly_diet_distribution(ax=ax3, average=average)

            ax4 = plt.subplot2grid((5, 4), (3, 0), colspan=2, rowspan=1)
            self.visualize_yearly_expense_distribution(ax=ax4, average=average)

            ax5 = plt.subplot2grid((5, 4), (3, 2), colspan=2, rowspan=1)
            self.visualize_monthly_expense_distribution(ax=ax5, average=average)
        else:
            ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=2)
            self.visualize_fund_distribution(ax=ax1)

            ax2 = plt.subplot2grid((5, 4), (2, 0), colspan=2, rowspan=1)
            self.visualize_investment_distribution(ax=ax2)

            ax3 = plt.subplot2grid((5, 4), (2, 2), colspan=2, rowspan=1)
            self.visualize_monthly_diet_distribution(ax=ax3)

            ax4 = plt.subplot2grid((5, 4), (3, 0), colspan=2, rowspan=1)
            self.visualize_yearly_expense_distribution(ax=ax4)

            ax5 = plt.subplot2grid((5, 4), (3, 2), colspan=2, rowspan=1)
            self.visualize_monthly_expense_distribution(ax=ax5)

        plt.tight_layout()
        plt.subplots_adjust(hspace=1.600, wspace=0.435, top=0.965, bottom=-0.100, left=0.045, right=0.985)
        plt.show()

    def visualize_fund_distribution(self, ax, month=None, average=False):
        if month:
            pass
        if average:
            pass

        colors = sns.color_palette("hls", n_colors=4)
        current_value = 0
        legend_lines = []
        for i in range(len(self.budget.fund_distribution)):
            label = self.budget.fund_distribution['Category'].values[i]
            height = self.budget.fund_distribution['Budget Amount'].values[i]
            if i == 0:
                bottom = 0
            else:
                bottom = current_value + height if height >= 0 else current_value

            color = colors[1] if height >= 0 else colors[0]
            bar = ax.bar(label, height, bottom=bottom, color=color)
            current_value += height
            self.visualizations[label] = mplcursors.cursor(bar, multiple=False, hover=True)
            self.visualizations[label].connect('add', self.get_budget_tooltip(bar=bar, fc=color, idx=i))
            legend_lines.append(mpatches.Patch(color=color, label=label))

        ax.set_xlabel('Category', fontsize=14)
        ax.set_ylabel('Budget Amount', fontsize=14)
        ax.set_title('Budget Waterfall Chart', fontsize=16)
        self.adjust_legend(legend_lines=legend_lines, ax=ax, colors=colors)
        ax.tick_params(axis='x', rotation=45)
        sns.despine()

    def visualize_monthly_expense_distribution(self, ax, month=None, average=False):
        # colors = sns.color_palette('cubehelix', len(self.budget.categorized_monthly_expenses.keys()))
        colors = sns.color_palette('Paired')
        budget_colors = colors[::2]
        spending_colors = colors[1::2]
        sorted_expenses = dict(sorted(self.budget.categorized_monthly_expenses.items(), key=lambda item: item[1], reverse=True))
        x = np.arange(len(sorted_expenses))

        if month:
            current_month = self.months[month]
            bars = ax.bar(x, [sorted_expenses[expense] for expense in sorted_expenses.keys()],
                          width=0.35, label='Monthly Budget')

            unique_spending_categories = [*self.actual_monthly_categorical_spending[((self.actual_monthly_categorical_spending['Date'] == month) & (self.actual_monthly_categorical_spending['Category'] != 'Income'))]['Category']]
            spending = {spending_category: self.actual_monthly_categorical_spending[((self.actual_monthly_categorical_spending['Category'] == spending_category) & (self.actual_monthly_categorical_spending['Date'] == month))]['Amount'].values[0] for spending_category in unique_spending_categories}
            for budget_category in sorted_expenses.keys():
                if budget_category not in spending.keys():
                    spending[budget_category] = 0

            bars2 = ax.bar(x + 0.35, [spending[spending_cat] for spending_cat in spending.keys()],
                           width=0.35, label=f'{current_month} Spending')
            legend_lines = []
            for idx, (bar, bar2) in enumerate(zip(bars, bars2)):
                budget_color = budget_colors[idx]
                spending_color = spending_colors[idx]
                bar.set_color(budget_color)
                bar2.set_color(spending_color)
                label = [*sorted_expenses.keys()][idx]
                # self.visualizations[label] = mplcursors.cursor(bar, multiple=False, hover=True)
                # self.visualizations[label].connect('add', self.get_monthly_expense_tooltip(bar=bar, fc=color, label=label))
                legend_lines.append(mpatches.Patch(color=budget_color, label=f'{label} Budget'))
                legend_lines.append(mpatches.Patch(color=spending_color, label=f'{label} Spending'))
        elif average:
            bars = ax.bar(x, [sorted_expenses[expense]['amount'] for expense in sorted_expenses.keys()],
                          width=0.35, label='Monthly Budget')

            bars2 = ax.bar(x + 0.35, [*self.average_monthly_categorical_spending.reset_index()['Amount']],
                           width=0.35, label='Monthly Average')
            legend_lines = []
            for idx, (bar, bar2, color) in enumerate(zip(bars, bars2, colors)):
                bar.set_color(color)
                label = [*sorted_expenses.keys()][idx]
                self.visualizations[label] = mplcursors.cursor(bar, multiple=False, hover=True)
                self.visualizations[label].connect('add',
                                                   self.get_monthly_expense_tooltip(bar=bar, fc=color, label=label))
                legend_lines.append(mpatches.Patch(color=color, label=label))
        else:
            bars = ax.bar([*sorted_expenses.keys()], [sorted_expenses[expense] for expense in sorted_expenses.keys()])
            bars2 = None
            legend_lines = []
            for idx, (bar, color) in enumerate(zip(bars, colors)):
                bar.set_color(color)
                label = [*sorted_expenses.keys()][idx]
                self.visualizations[label] = mplcursors.cursor(bar, multiple=False, hover=True)
                self.visualizations[label].connect('add',
                                                   self.get_monthly_expense_tooltip(bar=bar, fc=color, label=label))
                legend_lines.append(mpatches.Patch(color=color, label=label))

        ax.set_xlabel('Expense')
        ax.set_ylabel('Cost')
        ax.set_title('Monthly Expenses')
        self.adjust_legend(legend_lines=legend_lines, ax=ax, colors=colors)
        ax.tick_params(axis='x', rotation=45)

    def visualize_investment_distribution(self, ax, month=None, average=False):
        if month:
            pass
        if average:
            pass

        colors = sns.color_palette('coolwarm', len(self.budget.investment_portfolio.keys()))
        bars = ax.bar([*self.budget.investment_portfolio.keys()],
                      [self.budget.investment_portfolio[investment]['Monthly'] for investment in
                       self.budget.investment_portfolio.keys()])

        legend_lines = []
        for idx, (bar, color) in enumerate(zip(bars, colors)):
            bar.set_color(color)
            label = [*self.budget.investment_portfolio.keys()][idx]
            self.visualizations[label] = mplcursors.cursor(bar, multiple=False, hover=True)
            self.visualizations[label].connect('add', self.get_investment_tooltip(bar=bar, fc=color, label=label))
            legend_lines.append(mpatches.Patch(color=color, label=label))

        ax.set_xlabel('Stock Ticker')
        ax.set_ylabel('Portfolio Percentage')
        ax.set_title('Investment Portfolio')
        self.adjust_legend(legend_lines=legend_lines, ax=ax, colors=colors)
        ax.tick_params(axis='x', rotation=45)

    def visualize_monthly_diet_distribution(self, ax, month=None, average=False):
        if month:
            pass
        if average:
            pass

        colors = sns.color_palette('RdBu', len(self.budget.monthly_diet.keys()))
        sorted_diet = dict(sorted(self.budget.monthly_diet.items(), key=lambda item: item[1], reverse=True))
        bars = ax.bar([*sorted_diet.keys()], [sorted_diet[food] for food in sorted_diet.keys()])
        legend_lines = []
        for idx, (bar, color) in enumerate(zip(bars, colors)):
            bar.set_color(color)
            label = [*sorted_diet.keys()][idx]
            self.visualizations[label] = mplcursors.cursor(bar, multiple=False, hover=True)
            self.visualizations[label].connect('add', self.get_diet_tooltip(bar=bar, fc=color, label=label))
            legend_lines.append(mpatches.Patch(color=color, label=label))

        ax.set_xlabel('Food')
        ax.tick_params(axis='x', labelsize=5)
        ax.set_ylabel('Cost')
        ax.set_title('Monthly Diet')
        self.adjust_legend(legend_lines=legend_lines, ax=ax, colors=colors)
        ax.tick_params(axis='x', rotation=45)

    def visualize_yearly_expense_distribution(self, ax, month=None, average=False):
        if month:
            pass
        if average:
            pass

        colors = sns.color_palette('viridis', len(self.budget.yearly_expenses.keys()))
        sorted_expenses = dict(sorted(self.budget.yearly_expenses.items(), key=lambda item: item[1]['amount'], reverse=True))
        bars = ax.bar([*sorted_expenses.keys()], [sorted_expenses[expense]['amount'] for expense in sorted_expenses.keys()])
        legend_lines = []
        for idx, (bar, color) in enumerate(zip(bars, colors)):
            bar.set_color(color)
            label = [*sorted_expenses.keys()][idx]
            self.visualizations[label] = mplcursors.cursor(bar, multiple=False, hover=True)
            self.visualizations[label].connect('add', self.get_yearly_expense_tooltip(bar=bar, fc=color, label=label))
            legend_lines.append(mpatches.Patch(color=color, label=label))

        ax.set_xlabel('Yearly Expense')
        ax.set_ylabel('Cost')
        ax.set_title('Yearly Expenses')
        self.adjust_legend(legend_lines=legend_lines, ax=ax, colors=colors)
        ax.tick_params(axis='x', rotation=45)

    @staticmethod
    def set_annotation_visualization_features(sel, fc):
        sel.annotation.get_bbox_patch().set(fc=fc, alpha=0.5)
        sel.annotation.arrow_patch.set(color="black", alpha=0.2, arrowstyle="->",
                                       connectionstyle="arc3,rad=0.3")
        sel.annotation.draggable(state=True)

    def mouse_motion_handler(self, event):
        self.mouse_x = event.xdata
        self.mouse_y = event.ydata

    def get_budget_tooltip(self, bar, fc, idx):
        def func(sel):
            sel.annotation.set_text('')
            sel.annotation.get_bbox_patch().set(alpha=0.001)
            sel.annotation.arrow_patch.set(color="black", alpha=0.001, arrowstyle="->",
                                           connectionstyle="arc3,rad=0.3")
            if sel.target is not None:
                x, y = self.mouse_x, self.mouse_y
                bbox = bar.patches[0].get_bbox()
                if bbox.contains(x, y):
                    label = self.budget.fund_distribution['Category'].values[idx]
                    quantity = self.budget.fund_distribution['Budget Amount'].values[idx]
                    sel.annotation.set_text(f'{label} - {quantity}')
                    self.set_annotation_visualization_features(sel=sel, fc=fc)
                    return

        return func

    def get_investment_tooltip(self, bar, fc, label):
        def func(sel):
            sel.annotation.set_text('')
            sel.annotation.get_bbox_patch().set(alpha=0.001)
            sel.annotation.arrow_patch.set(color="black", alpha=0.001, arrowstyle="->",
                                           connectionstyle="arc3,rad=0.3")
            if sel.target is not None:
                x, y = self.mouse_x, self.mouse_y
                bbox = bar.get_bbox()
                if bbox.contains(x, y):
                    sel.annotation.set_text(f'{label} - {self.budget.investment_portfolio[label]["Monthly"]}')
                    self.set_annotation_visualization_features(sel=sel, fc=fc)
                    return

        return func

    def get_diet_tooltip(self, bar, fc, label):
        def func(sel):
            sel.annotation.set_text('')
            sel.annotation.get_bbox_patch().set(alpha=0.001)
            sel.annotation.arrow_patch.set(color="black", alpha=0.001, arrowstyle="->",
                                           connectionstyle="arc3,rad=0.3")
            if sel.target is not None:
                x, y = self.mouse_x, self.mouse_y
                bbox = bar.get_bbox()
                if bbox.contains(x, y):
                    sel.annotation.set_text(f'{label} - {self.budget.monthly_diet[label]}')
                    self.set_annotation_visualization_features(sel=sel, fc=fc)
                    return

        return func

    def get_monthly_expense_tooltip(self, bar, fc, label):
        def func(sel):
            sel.annotation.set_text('')
            sel.annotation.get_bbox_patch().set(alpha=0.001)
            sel.annotation.arrow_patch.set(color="black", alpha=0.001, arrowstyle="->",
                                           connectionstyle="arc3,rad=0.3")
            if sel.target is not None:
                x, y = self.mouse_x, self.mouse_y
                bbox = bar.get_bbox()
                if bbox.contains(x, y):
                    sel.annotation.set_text(f'{label} - {self.budget.monthly_expenses[label]["amount"]}')
                    self.set_annotation_visualization_features(sel=sel, fc=fc)
                    return

        return func

    def get_yearly_expense_tooltip(self, bar, fc, label):
        def func(sel):
            sel.annotation.set_text('')
            sel.annotation.get_bbox_patch().set(alpha=0.001)
            sel.annotation.arrow_patch.set(color="black", alpha=0.001, arrowstyle="->",
                                           connectionstyle="arc3,rad=0.3")
            if sel.target is not None:
                x, y = self.mouse_x, self.mouse_y
                bbox = bar.get_bbox()
                if bbox.contains(x, y):
                    sel.annotation.set_text(f'{label} - {self.budget.yearly_expenses[label]["amount"]}')
                    self.set_annotation_visualization_features(sel=sel, fc=fc)
                    return

        return func

    def calculate_statistics(self):
        self.comparison['Difference'] = self.comparison['Amount_Actual'] - self.comparison['Amount_Budgeted']
        mean_difference = self.comparison['Difference'].mean()
        std_difference = self.comparison['Difference'].std()
        return mean_difference, std_difference

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(X=self.actual_spending['Date'].values.reshape(-1, 1), y=self.actual_spending['Amount'])

    def predict_future_spending(self):
        if self.model is None:
            print("Model is not trained yet. Call train_model method first.")
            return

        future_dates = pd.date_range(start=self.start_date, periods=self.periods, freq='M')
        predicted_spending = self.model.predict(
            X=np.array(future_dates.to_series().apply(lambda x: x.toordinal()).tolist()).reshape(-1, 1))

        return predicted_spending


def create_monthly_budget(monthly_income: float, monthly_expenses: dict, monthly_diet: dict, yearly_expenses: dict, investment_portfolio: dict):
    budget = Budget(monthly_income=monthly_income,
                    monthly_expenses=monthly_expenses,
                    monthly_diet=monthly_diet,
                    yearly_expenses=yearly_expenses,
                    investment_portfolio=investment_portfolio)
    return budget


def budget_analyzer(monthly_expenses: dict,
                    yearly_expenses: dict,
                    monthly_diet: dict,
                    investment_portfolio: dict,
                    monthly_income: float,
                    start_date: str,
                    actual_spending: str):

    budget = create_monthly_budget(monthly_income=monthly_income,
                                   monthly_expenses=monthly_expenses,
                                   monthly_diet=monthly_diet,
                                   yearly_expenses=yearly_expenses,
                                   investment_portfolio=investment_portfolio)

    analyzed_budget = BudgetAnalyzer(budget=budget, start_date=start_date, actual_spending=actual_spending)
    # mean_difference, std_difference = analyzed_budget.calculate_statistics()
    # analyzed_budget.train_model()
    # predicted_spending = analyzed_budget.predict_future_spending()
    # print(f"Mean difference: {mean_difference}, Std deviation: {std_difference}")
    # print(f"Predicted spending: {predicted_spending}")
    return analyzed_budget


if __name__ == '__main__':
    bofa_monthly_expenses = {'AmazonPrime': {'amount': 15.13, 'day': 22, 'category': 'Entertainment'},
                             'Apple': {'amount': 2.99, 'day': 5, 'category': 'Utility'},
                             'Crunchyroll': {'amount': 7.99, 'day': 20, 'category': 'Entertainment'},
                             'JetBrains': {'amount': 6.90, 'day': 14, 'category': 'Education'},
                             'Spotify': {'amount': 18.09, 'day': 13, 'category': 'Entertainment'},
                             'FPL': {'amount': 180.45, 'day': 1, 'category': 'Utility'},
                             'Verizon': {'amount': 70.00, 'day': 17, 'category': 'Utility'},
                             'AT&T': {'amount': 62.31, 'day': 12, 'category': 'Utility'},
                             'ChatGPT': {'amount': 20.00, 'day': 16, 'category': 'Education'},
                             'CapitalOne': {'amount': 1000.00, 'day': 5, 'category': 'Transportation'},
                             'ShonenJump': {'amount': 1.99, 'day': 31, 'category': 'Entertainment'},
                             'Rent': {'amount': 3300.00, 'day': 1, 'category': 'Shelter'}
                             }
    bofa_yearly_expenses = {'Chess': {'amount': 29.99, 'date': '07-19', 'category': 'Entertainment'},
                            'MicrosoftOffice': {'amount': 99.99, 'date': '03-03', 'category': 'Utilities'},
                            'Ring': {'amount': 39.99, 'date': '10-06', 'category': 'Utilities'},
                            'Nintendo': {'amount': 19.99, 'date': '01-01', 'category': 'Entertainment'},
                            'PlayStation': {'amount': 59.99, 'date': '01-01', 'category': 'Entertainment'},
                            'MedicalMarijuana': {'amount': 400.00, 'date': '01-01', 'category': 'Utilities'},
                            'BristolWest': {'amount': 4732.00, 'date': '01-01', 'category': 'Transportation'},
                            }
    bofa_monthly_diet = {'Eggs': 60,
                         'Spinach': 16,
                         'Oil': 20,
                         'Bread': 25,
                         'Butter': 18,
                         'Pure Encapsulations Multivitamin': 40,
                         'Pure Encapsulations Vitamin D3': 30,
                         'Chobani Greek Yogurt': 30,
                         'Mandarin': 10,
                         'Banana': 20,
                         'Blueberries': 40,
                         'Apple': 56,
                         'Salmon': 200,
                         'Chicken': 250,
                         'Steak': 100,
                         'Quinoa / Rice': 40,
                         'Tomato': 50,
                         'Broccoli': 20,
                         'Cucumber': 30,
                         'Avocado': 60,
                         'Carrots': 16,
                         'Celery': 8,
                         }
    bofa_investment_portfolio = {'BTC': 0.250000,
                                 'ETH': 0.250000,
                                 'SPY': 0.140625,
                                 'SMH': 0.125000,
                                 'IWM': 0.109375,
                                 'EFA': 0.062500,
                                 'IYR': 0.046875,
                                 'GLD': 0.015625,
                                 }
    bofa_monthly_income = 9620.44
    bofa_start_date = '2023-01-01'
    bofa_actual_spending = r'C:\Users\Amram\IMPORTANT\Projects\dev\budget_dev\data\2023_transactions.csv'
    spending_analytics = budget_analyzer(monthly_expenses=bofa_monthly_expenses,
                                         yearly_expenses=bofa_yearly_expenses,
                                         monthly_diet=bofa_monthly_diet,
                                         investment_portfolio=bofa_investment_portfolio,
                                         monthly_income=bofa_monthly_income,
                                         start_date=bofa_start_date,
                                         actual_spending=bofa_actual_spending)
