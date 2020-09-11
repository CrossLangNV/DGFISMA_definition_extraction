from sklearn.metrics import classification_report, f1_score


def report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print(report)

    return report
