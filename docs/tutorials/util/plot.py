import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_pacf


def plot(y, model_result, suptitle=''):

    rmse = mean_squared_error(y, model_result.Z) ** 0.5
    r2 = r2_score(y, model_result.Z)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{suptitle}', fontsize=12, fontweight='bold')

    ax[0, 0].plot(y.index, model_result.Z, label='Ti Modelled', alpha=0.75)
    ax[0, 0].plot(y, label='Ti Measured', alpha=0.75)
    if hasattr(model_result, 'Te'):
        ax[0, 0].plot(y.index, model_result.Te, label='Te Modelled', alpha=0.75)
    ax[0, 0].legend()
    ax[0, 0].set_ylabel('[˚C]')
    if abs(rmse) < 100:
        ax[0, 0].set_title(f'RMSE: {rmse:.4f}')
    else:
        ax[0, 0].set_title(f'RMSE: {rmse:.4e}')
    plt.setp(ax[0, 0].get_xticklabels(), rotation=30, horizontalalignment='right')

    ax[1, 0].scatter(model_result.Z, y, alpha=0.2)
    ax[1, 0].set_xlabel('Ti Modelled [˚C]')
    ax[1, 0].set_ylabel('Ti Measured [˚C]')
    if abs(r2) < 100:
        ax[1, 0].set_title(f'$R^2$: {r2:.4f}')
    else:
        ax[1, 0].set_title(f'$R^2$: {r2:.4e}')
    plt.setp(ax[0, 1].get_xticklabels(), rotation=30, horizontalalignment='right')

    ax[0, 1].plot(y.index, y - model_result.Z, label='Ti Residual', color='black', alpha=0.75)
    ax[0, 1].legend()
    ax[0, 1].set_ylabel('[˚C]')
    ax[0, 1].set_title('Residuals')

    plot_pacf(y - model_result.Z, ax=ax[1, 1], lags=50)

    fig.tight_layout()


def plot_input_data(df):

    # keep this out of pandas plotting capabilities for compatibility
    # with the rest of the plots
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    ax[0].plot(df.index, df['Ti'], label='Ti', alpha=0.75)
    ax[0].plot(df.index, df['Ta'], label='Ta', alpha=0.75)
    ax[0].legend()
    ax[0].set_ylabel('[˚C]')
    ax[0].set_xticks([], minor=[])
    ax[0].set_xlim(df.index[0], df.index[-1])

    ax[1].plot(df.index, df['Ph'], label='Ph', alpha=0.75, color='black')
    ax[1].legend()
    ax[1].set_ylabel('[kWh]')
    ax[1].set_xlim(df.index[0], df.index[-1])

    fig.tight_layout()
